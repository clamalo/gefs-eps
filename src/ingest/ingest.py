import asyncio
import json
import aiohttp
import pandas as pd
import aiofiles
from tqdm.asyncio import tqdm as atqdm
from io import StringIO
import os
import multiprocessing
import logging
from asyncio import Semaphore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Define a global timeout value (in seconds)
GLOBAL_TIMEOUT = 60  # Adjust as needed

# Adaptive Semaphore Manager
class AdaptiveSemaphore:
    def __init__(self, initial_limit, max_limit, min_limit=5):
        self._semaphore = Semaphore(initial_limit)
        self._lock = asyncio.Lock()
        self.limit = initial_limit
        self.max_limit = max_limit
        self.min_limit = min_limit

    async def acquire(self):
        await self._semaphore.acquire()

    async def release(self):
        self._semaphore.release()

    async def increase_limit(self, factor=1):
        async with self._lock:
            new_limit = self.limit + factor
            if new_limit > self.max_limit:
                new_limit = self.max_limit
            for _ in range(factor):
                self._semaphore.release()
            self.limit = new_limit
            # logging.info(f"Semaphore limit increased to {self.limit}")

    async def decrease_limit(self, factor=1):
        async with self._lock:
            if self.limit - factor < self.min_limit:
                new_limit = self.min_limit
                factor = self.limit - self.min_limit
            else:
                new_limit = self.limit - factor
            for _ in range(factor):
                await self._semaphore.acquire()
            self.limit = new_limit
            logging.info(f"Semaphore limit decreased to {self.limit}")

    def current_limit(self):
        return self.limit

async def fetch(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()

async def parse_idx(session, idx_link):
    r = await fetch(session, idx_link)
    idx = [json.loads(line) for line in r.splitlines()]
    idx_df = pd.DataFrame(idx)
    idx_df.drop(columns=["domain", "stream", "date", "time", "expver", "class", "step"], inplace=True)
    idx_df["member"] = idx_df["number"].apply(lambda x: f"p{int(x):02d}" if not pd.isna(x) else "c00")
    idx_df.drop(columns=["number"], inplace=True)
    idx_df["level"] = idx_df.apply(lambda x: x["levtype"] if x["levtype"] == "sfc" else x["levelist"] + " hPa", axis=1)
    idx_df.drop(columns=["levtype", "levelist"], inplace=True)
    idx_df.rename(columns={"_offset": "start"}, inplace=True)
    idx_df["end"] = idx_df["start"] + idx_df["_length"]
    return idx_df

async def download(session, url, start, end, sem, semaphore_manager, max_retries=5):
    headers = {"Range": f"bytes={start}-{end}"}
    attempt = 0
    backoff = 1  # initial backoff in seconds

    while attempt < max_retries:
        try:
            await sem.acquire()
            async with session.get(url, headers=headers) as response:
                if response.status in (200, 206):
                    content = await response.read()
                    await sem.release()
                    return content
                elif response.status == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = int(retry_after)
                    else:
                        wait_time = backoff
                    logging.warning(f"Rate limited. Retrying after {wait_time} seconds.")
                    await asyncio.sleep(wait_time)
                    backoff *= 2
                    attempt += 1
                    await sem.release()
                    await semaphore_manager.decrease_limit()
                elif 500 <= response.status < 600:
                    logging.warning(f"Server error {response.status}. Retrying after {backoff} seconds.")
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    attempt += 1
                    await sem.release()
                else:
                    response.raise_for_status()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logging.warning(f"Attempt {attempt + 1}: Error downloading {url} bytes {start}-{end}: {e}")
            await asyncio.sleep(backoff)
            backoff *= 2
            attempt += 1
            await sem.release()

    logging.error(f"Failed to download {url} bytes {start}-{end} after {max_retries} attempts")
    raise Exception(f"Failed to download {url} bytes {start}-{end} after {max_retries} attempts")

async def download_data(date, cycle, step, req_param, req_level, output_file, mode='wb', semaphore_manager=None, session=None):
    # ROOT = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
    ROOT = "https://data.ecmwf.int/forecasts"
    host = f"{ROOT}/{date}/{cycle}z/ifs/0p25/enfo"
    idx_link = f"{host}/{date}{cycle}0000-{step}h-enfo-ef.index"

    idx_df = await parse_idx(session, idx_link)
    if req_level is None:
        idx_df = idx_df[idx_df.param.isin(req_param)]
    else:
        idx_df = idx_df[(idx_df.param.isin(req_param)) & (idx_df.level.isin(req_level))]
    idx = idx_df.to_dict(orient="records")
    pgrb2_link = f"{host}/{date}{cycle}0000-{step}h-enfo-ef.grib2"

    tasks = [
        download(session, pgrb2_link, record["start"], record["end"], semaphore_manager, semaphore_manager)
        for record in idx
    ]

    with atqdm(total=len(tasks)) as pbar:
        async with aiofiles.open(output_file, mode) as f:
            for task in asyncio.as_completed(tasks):
                try:
                    content = await task
                    await f.write(content)
                    pbar.update(1)
                    # Optionally, increase semaphore limit on successful download
                    await semaphore_manager.increase_limit()
                except Exception as e:
                    logging.error(f"Error downloading chunk: {e}")

async def parse_gefs_idx(session, idx_link):
    r = await fetch(session, idx_link)
    data = StringIO(r)
    idx_df = pd.read_csv(data, sep=":", header=None)
    idx_df.columns = ["line", "start", "datestr", "param", "level", "forecast_step", "member"]
    idx_df["end"] = idx_df["start"].shift(-1).fillna(-1).astype(int)
    idx_df["link"] = idx_link.replace('.idx','')
    idx_df = idx_df[["line", "start", "end", "datestr", "param", "level", "forecast_step", "member", "link"]]
    idx_df.drop(columns=["datestr", "line", "forecast_step", "member"], inplace=True)
    return idx_df

async def download_gefs(date, cycle, step, req_param, req_level, output_file, mode='wb', semaphore_manager=None, session=None):
    ROOT = "https://noaa-gefs-pds.s3.amazonaws.com"
    members = [f"p{str(i).zfill(2)}" for i in range(1, 31)] + ["c00"]
    df_list = []

    tasks = []
    for member in members:
        if req_param == ['APCP']:
            host = f"{ROOT}/gefs.{date}/{cycle}/atmos/pgrb2sp25"
            idx_link = f"{host}/ge{member}.t{cycle}z.pgrb2s.0p25.f{step:03}.idx"
        else:
            host = f"{ROOT}/gefs.{date}/{cycle}/atmos/pgrb2ap5"
            idx_link = f"{host}/ge{member}.t{cycle}z.pgrb2a.0p50.f{step:03}.idx"
        tasks.append(parse_gefs_idx(session, idx_link))
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Error parsing GEFS index: {result}")
            else:
                df_list.append(result)
    except Exception as e:
        logging.error(f"Error parsing GEFS index: {e}")
        return

    if not df_list:
        logging.error("No index data available to download GEFS.")
        return

    idx_df = pd.concat(df_list, ignore_index=True)
    if req_level is None:
        idx_df = idx_df[idx_df.param.isin(req_param)]
    else:
        idx_df = idx_df[(idx_df.param.isin(req_param)) & (idx_df.level.isin(req_level))]
    idx = idx_df.to_dict(orient="records")

    download_tasks = [
        download(session, record['link'], record["start"], record["end"], semaphore_manager, semaphore_manager)
        for record in idx
    ]

    with atqdm(total=len(download_tasks)) as pbar:
        async with aiofiles.open(output_file, mode) as f:
            for task in asyncio.as_completed(download_tasks):
                try:
                    content = await task
                    await f.write(content)
                    pbar.update(1)
                    # Optionally, increase semaphore limit on successful download
                    await semaphore_manager.increase_limit()
                except Exception as e:
                    logging.error(f"Error downloading GEFS chunk: {e}")

async def ingest(date, cycle, step, gefs, eps):
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Determine initial semaphore limit based on CPU cores
    cpu_cores = multiprocessing.cpu_count()
    initial_limit = max(cpu_cores * 5, 10)  # Start with 2x CPU cores or at least 10
    max_limit = cpu_cores * 10  # Define a maximum limit to prevent overloading
    semaphore_manager = AdaptiveSemaphore(initial_limit=initial_limit, max_limit=max_limit)

    logging.info(f"Initialized semaphore with limit: {semaphore_manager.current_limit()}")

    # Configure a single ClientSession with limited connections
    connector = aiohttp.TCPConnector(limit_per_host=semaphore_manager.current_limit())
    timeout = aiohttp.ClientTimeout(total=GLOBAL_TIMEOUT)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        if eps:
            req_param = ["gh", "t"]
            req_level = ["1000 hPa", "925 hPa", "850 hPa", "700 hPa", "500 hPa"]
            await download_data(
                date, cycle, step, req_param, req_level,
                os.path.join(data_dir, f'ecmwf_{step}.grib2'), mode='wb',
                semaphore_manager=semaphore_manager,
                session=session
            )

            req_param = ["tp"]
            await download_data(
                date, cycle, step, req_param, None,
                os.path.join(data_dir, f'ecmwf_{step}.grib2'), mode='ab',
                semaphore_manager=semaphore_manager,
                session=session
            )

        if gefs:
            req_param = ["APCP"]
            await download_gefs(
                date, cycle, step, req_param, None,
                os.path.join(data_dir, f'gefs_{step}.grib2'), mode='wb',
                semaphore_manager=semaphore_manager,
                session=session
            )

            req_param = ["HGT", "TMP"]
            req_level = ["1000 mb", "925 mb", "850 mb", "700 mb", "500 mb"]
            await download_gefs(
                date, cycle, step, req_param, req_level,
                os.path.join(data_dir, f'p_gefs_{step}.grib2'), mode='wb',
                semaphore_manager=semaphore_manager,
                session=session
            )

# Example usage:
# asyncio.run(ingest('20240101', '00', '06', gefs=True, eps=True))