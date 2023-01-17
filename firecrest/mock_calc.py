"""A mock up how a calculation would be run in AiiDA with FirecREST.

See also https://firecrest-api.cscs.ch/
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
from pathlib import Path, PurePosixPath, PureWindowsPath
from pprint import pprint
from tempfile import TemporaryDirectory
from textwrap import dedent
import time
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse
from uuid import uuid4

import aiofiles
import aiohttp

import firecrest

LOGGER = logging.getLogger("aiida_calc")


@dataclass
class Computer:
    """A mock computer (including transport/scheduler)."""

    client_url: str
    # per-user authinfo
    client_id: str
    client_secret: str  # note this would not actually be stored in the database
    token_uri: str
    machine_name: str
    work_dir: str
    fsystem: Literal["posix", "windows"] = "posix"
    # decide whether a file can be uploaded directly,
    # over the REST API, or whether it needs to be uploaded
    small_file_size_mb: int = 5

    @property
    def work_path(self) -> PurePosixPath | PureWindowsPath:
        """Return the work directory path."""
        return (
            PurePosixPath(self.work_dir)
            if self.fsystem == "posix"
            else PureWindowsPath(self.work_dir)
        )

    @property
    def client(self) -> firecrest.Firecrest:
        """Return a FirecREST client.

        Cache the client instance, so that we don't have to re-authenticate
        (it automatically refreshes the token when it expires)
        """
        if not hasattr(self, "_client"):
            self._client = firecrest.Firecrest(
                firecrest_url=self.client_url,
                authorization=firecrest.ClientCredentialsAuth(
                    self.client_id, self.client_secret, self.token_uri
                ),
            )
        return self._client


@dataclass
class Data:
    """A mock data object."""

    uuid: str = field(default_factory=lambda: str(uuid4()))
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalcNode(Data):
    """A mock calculation data node."""


REMOTE_FOLDER_NAME = "aiida"


async def run_multiple_calculations(
    computer: Computer, calcs: list[CalcNode]
) -> dict[str, list[Data]]:
    """Run multiple calculations on a remote computer."""
    return {
        uid: nodes
        for uid, nodes in await asyncio.gather(
            *[run_calculation(computer, calc) for calc in calcs]
        )
    }


async def run_calculation(computer: Computer, calc: CalcNode):
    """Run a process on a remote computer."""

    with TemporaryDirectory() as in_tmpdir:
        in_tmpdir = Path(in_tmpdir)
        await prepare_for_submission(calc, in_tmpdir)
        await asyncio.sleep(0)
        await copy_to_remote(computer, calc, in_tmpdir)
        await asyncio.sleep(0)

    await submit_on_remote(computer, calc)
    await asyncio.sleep(0)

    await poll_until_finished(computer, calc)
    await asyncio.sleep(0)

    with TemporaryDirectory() as out_tmpdir:
        out_tmpdir = Path(out_tmpdir)
        await copy_from_remote(computer, calc, out_tmpdir)
        await asyncio.sleep(0)
        return calc.uuid, await parse_output_files(calc, out_tmpdir)


async def prepare_for_submission(calc: CalcNode, local_path: Path):
    """Prepares the (local) calculation folder with all inputs,
    ready to be copied to the compute resource.
    """
    LOGGER.info("prepare for submission: %s", calc.uuid)
    local_path.joinpath("job.sh").write_text(
        dedent(
            f"""\
        #!/bin/bash
        #SBATCH --job-name={calc.uuid}

        mkdir -p output
        echo "Hello world!" > output/output.txt
        """
        )
    )


async def poll_object_transfer(
    obj: firecrest.ExternalStorage, interval: int = 1, timeout: int | None = 60
):
    """Poll until an object  has been transferred to/from the store."""
    start = time.time()
    while obj.in_progress:
        if timeout and time.time() - start > timeout:
            raise RuntimeError("timeout waiting for object transfer")
        await asyncio.sleep(interval)


async def copy_to_remote(computer: Computer, calc: CalcNode, local_folder: Path):
    """Copy the calculation inputs to the compute resource."""
    remote_folder = computer.work_path / REMOTE_FOLDER_NAME / calc.uuid
    LOGGER.info("copying to remote folder: %s", remote_folder)
    client = computer.client
    client.mkdir(computer.machine_name, str(remote_folder), p=True)
    for local_path in local_folder.glob("**/*"):
        target_path = remote_folder.joinpath(
            *local_path.relative_to(local_folder).parts
        )
        LOGGER.debug("copying to remote: %s", target_path)
        if local_path.is_dir():
            client.mkdir(computer.machine_name, str(target_path), p=True)
        if local_path.is_file():
            if computer.small_file_size_mb * 1024 * 1024 > local_path.stat().st_size:
                client.simple_upload(
                    computer.machine_name, str(local_path), str(target_path.parent)
                )
                await asyncio.sleep(0)
            else:
                up_obj = client.external_upload(
                    computer.machine_name, str(local_path), str(target_path.parent)
                )
                # TODO here we do not use pyfirecrest's finish_upload,
                # since it simply runs a subprocess to do the upload (calling curl)
                # instead we properly async the upload
                # up_obj.finish_upload()
                params = up_obj.object_storage_data["parameters"]
                # TODO this local fix for MACs was necessary for the demo
                params["url"] = params["url"].replace("192.168.220.19", "localhost")
                await upload_file_to_url(local_path, params)
                await poll_object_transfer(up_obj)


async def submit_on_remote(computer: Computer, calc: CalcNode):
    """Run the calculation on the compute resource."""
    script_path = computer.work_path / REMOTE_FOLDER_NAME / calc.uuid / "job.sh"
    LOGGER.info("submitting on remote: %s", script_path)
    client = computer.client
    result = client.submit(computer.machine_name, str(script_path), local_file=False)
    calc.attributes["job_id"] = result["jobid"]


async def poll_until_finished(
    computer: Computer, calc: CalcNode, interval: int = 1, timeout: int | None = 60
):
    """Poll the compute resource until the calculation is finished."""
    LOGGER.info("polling job until finished: %s", calc.uuid)
    client = computer.client
    start = time.time()
    while timeout is None or (time.time() - start) < timeout:
        results = client.poll(computer.machine_name, [calc.attributes["job_id"]])
        if results and results[0]["state"] == "COMPLETED":
            break
        await asyncio.sleep(interval)
    else:
        raise RuntimeError("timeout waiting for calculation to finish")


async def copy_from_remote(computer: Computer, calc: CalcNode, local_folder: Path):
    """Copy the calculation outputs from the compute resource."""
    remote_folder = computer.work_path / REMOTE_FOLDER_NAME / calc.uuid
    LOGGER.info("copying from remote folder: %s", remote_folder)
    client = computer.client
    for item in client.ls_recurse(
        computer.machine_name, str(remote_folder), show_hidden=True
    ):
        if item["type"] == "-":
            remote_path = remote_folder / item["path"]
            LOGGER.debug("copying from remote: %s", remote_path)
            local_path = local_folder.joinpath(
                *remote_path.relative_to(remote_folder).parts
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if computer.small_file_size_mb * 1024 * 1024 > int(item["size"]):
                client.simple_download(
                    computer.machine_name, str(remote_path), str(local_path)
                )
                await asyncio.sleep(0)
            else:
                down_obj = client.external_download(
                    computer.machine_name, str(remote_path)
                )
                await poll_object_transfer(down_obj)

                # TODO here instead of using down_obj.finish_download
                # we use an asynchoronous version of it
                url = down_obj.object_storage_data
                # await download_url_to_file(url, local_path)

                # TODO however the url above doesn't work locally, with the demo docker
                # there was a fix already noted for MAC: url.replace("192.168.220.19", "localhost")
                # however, this still gives a 403 error:
                # "The request signature we calculated does not match the signature you provided.
                # Check your key and signing method.""
                # so for now, I'm just going to swap out the URL, with the actual location on disk
                # where the files are stored for the demo!
                store_path = (
                    "/Users/chrisjsewell/Documents/GitHub/firecrest/deploy/demo/minio"
                    + urlparse(url).path
                )
                await copy_file_async(store_path, local_path)


async def parse_output_files(calc: CalcNode, local_path: Path) -> list[Data]:
    """Parse the calculation outputs."""
    LOGGER.info("parsing output files: %s", local_path)
    paths = []
    for path in local_path.glob("**/*"):
        paths.append(
            path.relative_to(local_path).as_posix() + ("/" if path.is_dir() else "")
        )
    return [Data(attributes={"paths": paths})]


# HELPER functions


async def copy_file_async(src: str | Path, dest: str | Path):
    """Copy a file asynchronously."""
    async with aiofiles.open(src, mode="rb") as fr, aiofiles.open(
        dest, mode="wb"
    ) as fw:
        while True:
            chunk = await fr.read(1024)
            if not chunk:
                break
            await fw.write(chunk)


class UploadParameters(TypedDict):
    """Parameters for the calculation."""

    url: str
    method: str
    data: dict[str, str]
    headers: dict[str, str]
    json: dict
    params: dict[str, str]


async def upload_file_to_url(filepath: Path | str, params: UploadParameters) -> str:
    """Upload a file from a local file to a URL."""
    # assert params["method"] == "POST" and not params["json"]
    async with aiohttp.ClientSession() as session:
        with open(filepath, "rb") as f:
            form = aiohttp.FormData()
            form.add_field("file", f, filename=str(filepath))
            for key, value in params["data"].items():
                form.add_field(key, value)
            async with session.post(
                params["url"],
                data=form,
                headers=params["headers"],
                params=params["params"],
            ) as resp:
                return await resp.text()


async def download_url_to_file(url: str, filepath: Path | str):
    """Download a file from a URL to a local file."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            with open(filepath, "wb") as f:
                while True:
                    chunk = await resp.content.read(1024)
                    if not chunk:
                        break
                    f.write(chunk)


def main():
    """Run the example."""
    logging.basicConfig(
        format="%(asctime)s:%(name)s:%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )

    computer = Computer(
        client_url="http://localhost:8000/",
        client_id="firecrest-sample",
        client_secret="b391e177-fa50-4987-beaf-e6d33ca93571",
        token_uri="http://localhost:8080/auth/realms/kcrealm/protocol/openid-connect/token",
        machine_name="cluster",
        work_dir="/home/service-account-firecrest-sample",
        small_file_size_mb=0,
    )

    calc1 = CalcNode()
    calc2 = CalcNode()
    nodes = asyncio.run(run_multiple_calculations(computer, [calc1, calc2]))
    pprint(nodes)

    # TODO how to remove files from the object store?


if __name__ == "__main__":
    main()
