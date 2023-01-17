"""A mock up how a calculation would be run in AiiDA with FirecREST.

See https://firecrest-api.cscs.ch/
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import TemporaryDirectory
from textwrap import dedent
import time
from typing import Any, Literal
from uuid import uuid4
import logging

import firecrest

LOGGER = logging.getLogger("firecrest.mock_calc")


@dataclass
class Computer:
    """A mock computer."""

    client_url: str
    # per-user authinfo
    client_id: str
    client_secret: str  # note this would not actually be stored in the database
    token_uri: str
    machine_name: str
    work_dir: str
    # decide whether a file can be uploaded directly,
    # over the REST API, or whether it needs to be uploaded
    small_file_size_mb: int = 5
    fsystem: Literal["posix", "windows"] = "posix"

    @property
    def work_path(self) -> PurePosixPath | PureWindowsPath:
        """Return the work directory path."""
        return (
            PurePosixPath(self.work_dir)
            if self.fsystem == "posix"
            else PureWindowsPath(self.work_dir)
        )


@dataclass
class Data:
    """A mock data object."""

    uuid: str = field(default_factory=lambda: str(uuid4()))
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalcNode(Data):
    """A mock calculation data node."""


REMOTE_FOLDER_NAME = "aiida"


def run_calculation(computer: Computer, calc: CalcNode):
    """Run a process on a remote computer."""

    with TemporaryDirectory() as in_tmpdir:
        in_tmpdir = Path(in_tmpdir)
        prepare_for_submission(calc, in_tmpdir)
        copy_to_remote(computer, calc, in_tmpdir)

    submit_on_remote(computer, calc)

    poll_until_finished(computer, calc)

    with TemporaryDirectory() as out_tmpdir:
        out_tmpdir = Path(out_tmpdir)
        copy_from_remote(computer, calc, out_tmpdir)
        return parse_output_files(calc, out_tmpdir)


def prepare_for_submission(calc: CalcNode, local_path: Path):
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


def copy_to_remote(computer: Computer, calc: CalcNode, local_path: Path):
    """Copy the calculation inputs to the compute resource."""
    remote_folder = computer.work_path / REMOTE_FOLDER_NAME / calc.uuid
    LOGGER.info("copying to remote folder: %s", remote_folder)
    client = _client_from_computer(computer)
    client.mkdir(computer.machine_name, str(remote_folder), p=True)
    for path in local_path.glob("**/*"):
        target_path = remote_folder.joinpath(*path.relative_to(local_path).parts)
        LOGGER.debug("copying to remote: %s", target_path)
        if path.is_dir():
            client.mkdir(computer.machine_name, str(target_path), p=True)
        if path.is_file():
            if computer.small_file_size_mb * 1024 * 1024 > path.stat().st_size:
                client.simple_upload(
                    computer.machine_name, str(path), str(target_path.parent)
                )
            else:
                up_obj = client.external_upload(
                    computer.machine_name, str(path), str(target_path.parent)
                )
                up_obj.finish_upload()
                start = time.time()
                while up_obj.in_progress:
                    if time.time() - start > 60:
                        raise RuntimeError(
                            f"timeout waiting for upload to finish of: {target_path}"
                        )
                    time.sleep(1)


def submit_on_remote(computer: Computer, calc: CalcNode):
    """Run the calculation on the compute resource."""
    script_path = computer.work_path / REMOTE_FOLDER_NAME / calc.uuid / "job.sh"
    LOGGER.info("submitting on remote: %s", script_path)
    client = _client_from_computer(computer)
    result = client.submit(computer.machine_name, str(script_path), local_file=False)
    calc.attributes["job_id"] = result["jobid"]


def poll_until_finished(computer: Computer, calc: CalcNode):
    """Poll the compute resource until the calculation is finished."""
    LOGGER.info("polling until finished: %s", calc.uuid)
    client = _client_from_computer(computer)
    start = time.time()
    while time.time() - start < 60:
        result = client.poll(computer.machine_name, [calc.attributes["job_id"]])[0]
        if result["state"] == "COMPLETED":
            break
    else:
        raise RuntimeError("timeout waiting for calculation to finish")


def copy_from_remote(computer: Computer, calc: CalcNode, local_folder: Path):
    """Copy the calculation outputs from the compute resource."""
    remote_folder = computer.work_path / REMOTE_FOLDER_NAME / calc.uuid
    LOGGER.info("copying from remote folder: %s", remote_folder)
    client = _client_from_computer(computer)
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
            else:
                down_obj = client.external_download(
                    computer.machine_name, str(remote_path)
                )
                down_obj.object_storage_data
                down_obj.finish_download(str(local_path))


def parse_output_files(calc: CalcNode, local_path: Path) -> list[Data]:
    """Parse the calculation outputs."""
    LOGGER.info("parsing output files: %s", local_path)
    paths = []
    for path in local_path.glob("**/*"):
        paths.append(
            path.relative_to(local_path).as_posix() + ("/" if path.is_dir() else "")
        )
    return [calc, Data(attributes={"paths": paths})]


@lru_cache(maxsize=256)
def _client_auth_from_computer(
    client_id: str, client_secret: str, token_uri: str
) -> firecrest.ClientCredentialsAuth:
    """Create a ClientCredentialsAuth from a Computer object.

    Note we cache this, because it retrieves an access token,
    which we don't want to do every time.
    (it automatically refreshes the token when it expires)
    """
    return firecrest.ClientCredentialsAuth(client_id, client_secret, token_uri)


def _client_from_computer(computer: Computer) -> firecrest.Firecrest:
    """Create a FireClient from a Computer object."""
    auth = _client_auth_from_computer(
        computer.client_id, computer.client_secret, computer.token_uri
    )
    return firecrest.Firecrest(firecrest_url=computer.client_url, authorization=auth)


def main():

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
        small_file_size_mb=5,
    )

    calc = CalcNode()
    nodes = run_calculation(computer, calc)
    print(nodes)


if __name__ == "__main__":
    main()
