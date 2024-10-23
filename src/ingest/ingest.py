import asyncio
import json
import aiohttp
import pandas as pd
import aiofiles
from tqdm.asyncio import tqdm as atqdm
from io import StringIO
import os
import time

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

async def download(session, url, start, end, sem, max_retries=5):
    headers = {"Range": f"bytes={start}-{end}"}
    attempt = 0
    backoff = 1  # initial backoff in seconds

    while attempt < max_retries:
        try:
            async with sem:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200 or response.status == 206:
                        return await response.read()
                    elif response.status == 429:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            wait_time = int(retry_after)
                        else:
                            wait_time = backoff
                        await asyncio.sleep(wait_time)
                        backoff *= 2
                        attempt += 1
                    elif 500 <= response.status < 600:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        attempt += 1
                    else:
                        response.raise_for_status()
        except aiohttp.ClientError as e:
            await asyncio.sleep(backoff)
            backoff *= 2
            attempt += 1

    raise Exception(f"Failed to download {url} after {max_retries} attempts")

async def download_data(date, cycle, step, req_param, req_level, output_file, mode='wb'):
    ROOT = "https://ai4edataeuwest.blob.core.windows.net/ecmwf"
    host = f"{ROOT}/{date}/{cycle}z/ifs/0p25/enfo"
    idx_link = f"{host}/{date}{cycle}0000-{step}h-enfo-ef.index"

    sem = asyncio.Semaphore(20)  # Limit to 20 concurrent downloads

    async with aiohttp.ClientSession() as session:
        idx_df = await parse_idx(session, idx_link)
        if req_level is None:
            idx_df = idx_df[(idx_df.param.isin(req_param))]
        else:
            idx_df = idx_df[(idx_df.param.isin(req_param)) & (idx_df.level.isin(req_level))]
        idx = idx_df.to_dict(orient="records")
        pgrb2_link = f"{host}/{date}{cycle}0000-{step}h-enfo-ef.grib2"

        tasks = [download(session, pgrb2_link, record["start"], record["end"], sem) for record in idx]

        with atqdm(total=len(tasks)) as pbar:
            async with aiofiles.open(output_file, mode) as f:
                for task in asyncio.as_completed(tasks):
                    try:
                        content = await task
                        await f.write(content)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error downloading chunk: {e}")

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

async def download_gefs(date, cycle, step, req_param, req_level, output_file, mode='wb'):
    ROOT = "https://noaa-gefs-pds.s3.amazonaws.com"
    members = [f"p{str(i).zfill(2)}" for i in range(1, 31)] + ["c00"]
    df_list = []

    sem = asyncio.Semaphore(5)  # Reduced concurrency to 5 for GEFS downloads

    async with aiohttp.ClientSession() as session:
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
            results = await asyncio.gather(*tasks)
            df_list.extend(results)
        except Exception as e:
            print(f"Error parsing GEFS index: {e}")
            return

        idx_df = pd.concat(df_list, ignore_index=True)
        if req_level is None:
            idx_df = idx_df[(idx_df.param.isin(req_param))]
        else:
            idx_df = idx_df[(idx_df.param.isin(req_param)) & (idx_df.level.isin(req_level))]
        idx = idx_df.to_dict(orient="records")

        download_tasks = [download(session, record['link'], record["start"], record["end"], sem) for record in idx]

        with atqdm(total=len(download_tasks)) as pbar:
            async with aiofiles.open(output_file, mode) as f:
                for task in asyncio.as_completed(download_tasks):
                    try:
                        content = await task
                        await f.write(content)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error downloading GEFS chunk: {e}")

async def ingest(date, cycle, step, gefs, eps):
    if eps:
        req_param = ["gh", "t"]
        req_level = ["1000 hPa", "925 hPa", "850 hPa", "700 hPa", "500 hPa"]
        await download_data(date, cycle, step, req_param, req_level, os.path.join(os.getcwd(), 'data', f'ecmwf_{step}.grib2'), mode='wb')

        req_param = ["tp"]
        await download_data(date, cycle, step, req_param, None, os.path.join(os.getcwd(), 'data', f'ecmwf_{step}.grib2'), mode='ab')

    if gefs:
        req_param = ["APCP"]
        await download_gefs(date, cycle, step, req_param, None, os.path.join(os.getcwd(), 'data', f'gefs_{step}.grib2'), mode='wb')

        req_param = ["HGT", "TMP"]
        req_level = ["1000 mb", "925 mb", "850 mb", "700 mb", "500 mb"]
        await download_gefs(date, cycle, step, req_param, req_level, os.path.join(os.getcwd(), 'data', f'p_gefs_{step}.grib2'), mode='wb')
