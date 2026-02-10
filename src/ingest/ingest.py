import asyncio
import json
import aiohttp
import pandas as pd
import aiofiles
from tqdm.asyncio import tqdm as atqdm
from io import StringIO
import os
import glob
import multiprocessing
import logging
from typing import Any, Dict, List, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# Define a global timeout value (in seconds)
GLOBAL_TIMEOUT = 60  # Adjust as needed


def _default_max_concurrency() -> int:
    # Network-bound, but too much concurrency triggers 429s and other instability.
    cpu_cores = multiprocessing.cpu_count()
    return min(48, max(8, cpu_cores * 2))


async def fetch(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()


async def parse_idx(session, idx_link):
    r = await fetch(session, idx_link)
    idx = [json.loads(line) for line in r.splitlines()]
    idx_df = pd.DataFrame(idx)
    idx_df.drop(
        columns=["domain", "stream", "date", "time", "expver", "class", "step"],
        inplace=True,
        errors="ignore",
    )
    idx_df["member"] = idx_df["number"].apply(
        lambda x: f"p{int(x):02d}" if not pd.isna(x) else "c00"
    )
    idx_df.drop(columns=["number"], inplace=True)
    idx_df["level"] = idx_df.apply(
        lambda x: x["levtype"] if x["levtype"] == "sfc" else x["levelist"] + " hPa",
        axis=1,
    )
    idx_df.drop(columns=["levtype", "levelist"], inplace=True)
    idx_df.rename(columns={"_offset": "start"}, inplace=True)
    # HTTP Range is inclusive; _length is a byte count.
    idx_df["end"] = idx_df["start"] + idx_df["_length"] - 1
    return idx_df


def _build_range_header(start: int, end: Optional[int]) -> Dict[str, str]:
    if end is None:
        return {"Range": f"bytes={start}-"}
    return {"Range": f"bytes={start}-{end}"}


async def download_range(
    session: aiohttp.ClientSession,
    url: str,
    start: int,
    end: Optional[int],
    sem: asyncio.Semaphore,
    *,
    max_retries: int = 5,
) -> bytes:
    attempt = 0
    backoff = 1.0
    headers = _build_range_header(start, end)

    while attempt < max_retries:
        await sem.acquire()
        try:
            async with session.get(url, headers=headers) as response:
                if response.status in (200, 206):
                    return await response.read()
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait_time = float(retry_after) if retry_after else backoff
                    logging.warning(
                        f"Rate limited (429). Retrying after {wait_time} seconds."
                    )
                    await asyncio.sleep(wait_time)
                    backoff *= 2
                    attempt += 1
                    continue
                if 400 <= response.status < 500:
                    # Permanent error (except 429 handled above).
                    response.raise_for_status()
                if 500 <= response.status < 600:
                    logging.warning(
                        f"Server error {response.status}. Retrying after {backoff} seconds."
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    attempt += 1
                    continue
                response.raise_for_status()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logging.warning(
                f"Attempt {attempt + 1}: Error downloading {url} bytes {start}-{end if end is not None else ''}: {e}"
            )
            await asyncio.sleep(backoff)
            backoff *= 2
            attempt += 1
        finally:
            sem.release()

    raise RuntimeError(
        f"Failed to download {url} bytes {start}-{end if end is not None else ''} after {max_retries} attempts"
    )


async def download_records_concat(
    session: aiohttp.ClientSession,
    records: List[Dict[str, Any]],
    output_file: str,
    sem: asyncio.Semaphore,
    *,
    description: str,
    max_concurrency: int,
) -> None:
    if not records:
        raise ValueError("No records to download.")

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{output_file}.part"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    queue = asyncio.Queue(maxsize=max(1, max_concurrency * 2))

    async def worker(i: int, record: Dict[str, Any]) -> None:
        content = await download_range(
            session,
            record["url"],
            int(record["start"]),
            record["end"],
            sem,
        )
        await queue.put((i, content))

    async def writer() -> None:
        pending: Dict[int, bytes] = {}
        next_i = 0
        with atqdm(total=len(records), desc=description) as pbar:
            async with aiofiles.open(tmp_path, "wb") as f:
                while next_i < len(records):
                    i, content = await queue.get()
                    pending[i] = content
                    while next_i in pending:
                        await f.write(pending.pop(next_i))
                        pbar.update(1)
                        next_i += 1

    writer_task = asyncio.create_task(writer())
    worker_tasks = [
        asyncio.create_task(worker(i, record)) for i, record in enumerate(records)
    ]
    results = await asyncio.gather(*worker_tasks, return_exceptions=True)
    first_exc = next((r for r in results if isinstance(r, Exception)), None)
    if first_exc is not None:
        for t in worker_tasks:
            t.cancel()
        writer_task.cancel()
        raise first_exc

    await writer_task
    os.replace(tmp_path, output_file)


def _remove_cfgrib_index_files(grib_path: str) -> None:
    for idx_path in glob.glob(f"{grib_path}.*.idx"):
        try:
            os.remove(idx_path)
        except OSError:
            pass


async def download_ecmwf_subset(
    session: aiohttp.ClientSession,
    date: str,
    cycle: str,
    step: int,
    req_param_any_level: Set[str],
    req_param_level_filtered: Set[str],
    req_levels: Set[str],
    output_file: str,
    sem: asyncio.Semaphore,
    max_concurrency: int,
    *,
    model: str,
    file_tag: str,
    description: str,
) -> None:
    root = "https://data.ecmwf.int/forecasts"
    host = f"{root}/{date}/{cycle}z/{model}/0p25/enfo"
    idx_link = f"{host}/{date}{cycle}0000-{step}h-{file_tag}.index"
    pgrb2_link = f"{host}/{date}{cycle}0000-{step}h-{file_tag}.grib2"

    idx_df = await parse_idx(session, idx_link)
    params = req_param_any_level | req_param_level_filtered
    idx_df = idx_df[idx_df.param.isin(params)]
    if req_param_level_filtered:
        mask_level_filtered = idx_df.param.isin(
            req_param_level_filtered
        ) & idx_df.level.isin(req_levels)
        mask_any_level = idx_df.param.isin(req_param_any_level)
        idx_df = idx_df[mask_level_filtered | mask_any_level]

    idx_df = idx_df.sort_values("start")
    records = [
        {"url": pgrb2_link, "start": int(r["start"]), "end": int(r["end"])}
        for r in idx_df.to_dict(orient="records")
    ]
    await download_records_concat(
        session,
        records,
        output_file,
        sem,
        description=description,
        max_concurrency=max_concurrency,
    )


async def parse_gefs_idx(session, idx_link):
    r = await fetch(session, idx_link)
    data = StringIO(r)
    idx_df = pd.read_csv(data, sep=":", header=None)
    idx_df.columns = [
        "line",
        "start",
        "datestr",
        "param",
        "level",
        "forecast_step",
        "member",
    ]
    idx_df["start"] = idx_df["start"].astype(int)
    idx_df["end"] = idx_df["start"].shift(-1) - 1
    idx_df["link"] = idx_link.replace(".idx", "")
    idx_df = idx_df[
        [
            "line",
            "start",
            "end",
            "datestr",
            "param",
            "level",
            "forecast_step",
            "member",
            "link",
        ]
    ]
    idx_df.drop(columns=["datestr", "line", "forecast_step", "member"], inplace=True)
    idx_df.loc[idx_df.index[-1], "end"] = None
    return idx_df


async def download_gefs_subset(
    session: aiohttp.ClientSession,
    date: str,
    cycle: str,
    step: int,
    req_param: Set[str],
    req_level: Optional[Set[str]],
    output_file: str,
    sem: asyncio.Semaphore,
    max_concurrency: int,
    *,
    description: str,
) -> None:
    root = "https://noaa-gefs-pds.s3.amazonaws.com"
    members = [f"p{str(i).zfill(2)}" for i in range(1, 31)] + ["c00"]

    idx_links = []
    for member in members:
        if req_param == {"APCP"}:
            host = f"{root}/gefs.{date}/{cycle}/atmos/pgrb2sp25"
            idx_links.append(f"{host}/ge{member}.t{cycle}z.pgrb2s.0p25.f{step:03}.idx")
        else:
            host = f"{root}/gefs.{date}/{cycle}/atmos/pgrb2ap5"
            idx_links.append(f"{host}/ge{member}.t{cycle}z.pgrb2a.0p50.f{step:03}.idx")

    idx_tasks = [
        asyncio.create_task(parse_gefs_idx(session, link)) for link in idx_links
    ]
    results = await asyncio.gather(*idx_tasks, return_exceptions=True)
    df_list: List[pd.DataFrame] = []
    errors: List[Exception] = []
    for r in results:
        if isinstance(r, Exception):
            errors.append(r)
        else:
            df_list.append(r)
    if errors:
        raise RuntimeError(
            f"Failed to parse {len(errors)} GEFS index files; first error: {errors[0]}"
        )
    if not df_list:
        raise RuntimeError("No GEFS index data available.")

    idx_df = pd.concat(df_list, ignore_index=True)
    idx_df = idx_df[idx_df.param.isin(req_param)]
    if req_level is not None:
        idx_df = idx_df[idx_df.level.isin(req_level)]

    idx_df = idx_df.sort_values(["link", "start"])
    records = []
    for r in idx_df.to_dict(orient="records"):
        end = r["end"]
        if pd.isna(end):
            end = None
        else:
            end = int(end)
        records.append({"url": r["link"], "start": int(r["start"]), "end": end})

    await download_records_concat(
        session,
        records,
        output_file,
        sem,
        description=description,
        max_concurrency=max_concurrency,
    )


async def ingest(date, cycle, step, gefs, eps, aifs, force=False, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    downloaded = {"eps": False, "aifs": False, "gefs": False, "p_gefs": False}

    max_concurrency = _default_max_concurrency()
    sem = asyncio.Semaphore(max_concurrency)
    logging.info(f"Using max download concurrency: {max_concurrency}")

    # Configure a single ClientSession with limited connections
    connector = aiohttp.TCPConnector(
        limit_per_host=max_concurrency, limit=max_concurrency * 2
    )
    timeout = aiohttp.ClientTimeout(total=GLOBAL_TIMEOUT)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        def has_local_file(path):
            return os.path.exists(path) and os.path.getsize(path) > 0

        if eps:
            eps_path = os.path.join(data_dir, f"ecmwf_{step}.grib2")
            if not force and has_local_file(eps_path):
                logging.info(
                    f"Skipping EPS download; found existing file at {eps_path}"
                )
            else:
                if force:
                    logging.info(f"Forcing EPS download; overwriting {eps_path}")
                _remove_cfgrib_index_files(eps_path)
                await download_ecmwf_subset(
                    session,
                    date,
                    cycle,
                    step,
                    req_param_any_level={"tp"},
                    req_param_level_filtered={"gh", "t"},
                    req_levels={"1000 hPa", "925 hPa", "850 hPa", "700 hPa", "500 hPa"},
                    output_file=eps_path,
                    sem=sem,
                    max_concurrency=max_concurrency,
                    model="ifs",
                    file_tag="enfo-ef",
                    description="Downloading EPS",
                )
                downloaded["eps"] = True

        if aifs:
            aifs_path = os.path.join(data_dir, f"aifs_{step}.grib2")
            if not force and has_local_file(aifs_path):
                logging.info(
                    f"Skipping AIFS download; found existing file at {aifs_path}"
                )
            else:
                if force:
                    logging.info(f"Forcing AIFS download; overwriting {aifs_path}")
                _remove_cfgrib_index_files(aifs_path)
                await download_ecmwf_subset(
                    session,
                    date,
                    cycle,
                    step,
                    req_param_any_level={"tp"},
                    req_param_level_filtered={"z", "t"},
                    req_levels={"1000 hPa", "925 hPa", "850 hPa", "700 hPa", "500 hPa"},
                    output_file=aifs_path,
                    sem=sem,
                    max_concurrency=max_concurrency,
                    model="aifs-ens",
                    file_tag="enfo-pf",
                    description="Downloading AIFS-E",
                )
                downloaded["aifs"] = True

        if gefs:
            gefs_path = os.path.join(data_dir, f"gefs_{step}.grib2")
            p_gefs_path = os.path.join(data_dir, f"p_gefs_{step}.grib2")
            if not force and has_local_file(gefs_path):
                logging.info(
                    f"Skipping GEFS APCP download; found existing file at {gefs_path}"
                )
            else:
                if force:
                    logging.info(f"Forcing GEFS APCP download; overwriting {gefs_path}")
                _remove_cfgrib_index_files(gefs_path)
                await download_gefs_subset(
                    session,
                    date,
                    cycle,
                    step,
                    req_param={"APCP"},
                    req_level=None,
                    output_file=gefs_path,
                    sem=sem,
                    max_concurrency=max_concurrency,
                    description="Downloading GEFS (APCP)",
                )
                downloaded["gefs"] = True

            if not force and has_local_file(p_gefs_path):
                logging.info(
                    f"Skipping GEFS pressure-level download; found existing file at {p_gefs_path}"
                )
            else:
                if force:
                    logging.info(
                        f"Forcing GEFS pressure-level download; overwriting {p_gefs_path}"
                    )
                _remove_cfgrib_index_files(p_gefs_path)
                await download_gefs_subset(
                    session,
                    date,
                    cycle,
                    step,
                    req_param={"HGT", "TMP"},
                    req_level={"1000 mb", "925 mb", "850 mb", "700 mb", "500 mb"},
                    output_file=p_gefs_path,
                    sem=sem,
                    max_concurrency=max_concurrency,
                    description="Downloading GEFS (pressure levels)",
                )
                downloaded["p_gefs"] = True

    return downloaded


# Example usage:
# asyncio.run(ingest('20240101', '00', '06', gefs=True, eps=True))
