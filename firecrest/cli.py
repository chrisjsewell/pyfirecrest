"""A simple CLI for pyfirecrest."""
from __future__ import annotations
from datetime import datetime

from pathlib import Path

import firecrest as fc
import json
import yaml
import click


class LazyConfig:
    """A lazy configuration object."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self._params = None
        self._client = None
        self._default_machine = None

    @property
    def params(self) -> dict:
        if self._params is None:
            with self.file_path.open("rb") as f:
                config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise TypeError("Config file must be a dictionary")
            required = {
                "client_id",
                "client_secret",
                "token_uri",
                "client_url",
                "default_machine",
            }
            missing = required - set(config.keys())
            if missing:
                raise ValueError(f"Config file missing keys: {missing}")
            self._params = config
        return self._params

    @property
    def client(self) -> fc.Firecrest:
        if self._client is None:
            config = self.params
            auth = fc.ClientCredentialsAuth(
                client_id=config["client_id"],
                client_secret=config["client_secret"],
                token_uri=config["token_uri"],
            )
            self._client = fc.Firecrest(
                firecrest_url=config["client_url"], authorization=auth
            )
            self._default_machine = config["default_machine"]

        return self._client

    @property
    def default_machine(self) -> str:
        return self.params["default_machine"]


pass_config = click.make_pass_decorator(LazyConfig, ensure=True)


@click.group()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    default=".fclient.yaml",
    help="Path to the configuration file",
    show_default=True,
)
@click.pass_context
def main(ctx: click.Context, config):
    """A simple CLI for pyfirecrest."""
    ctx.ensure_object(dict)
    ctx.obj = LazyConfig(config)


@main.command
@pass_config
def whoami(config: LazyConfig):
    """Print the username."""
    print(config.client.whoami())


@main.group()
def status():
    """`/status/` endpoints."""


@status.command()
@pass_config
def parameters(config: LazyConfig):
    """GET `/status/parameters`"""
    print(yaml.dump(config.client.parameters()))


@status.command()
@click.argument("name", required=False, type=str, default=None)
@pass_config
def services(config: LazyConfig, name: str | None):
    """GET `/status/services`"""
    if name:
        print(yaml.dump(config.client.service(name)))
    else:
        print(yaml.dump(config.client.all_services()))


@status.command()
@click.argument("name", required=False, type=str, default=None)
@pass_config
def systems(config: LazyConfig, name: str | None):
    """GET `/status/systems`"""
    if name:
        print(yaml.dump(config.client.system(name)))
    else:
        print(yaml.dump(config.client.all_systems()))


@main.group()
def util():
    """`/utilities/` endpoints."""


MACHINE_OPTION = click.option(
    "-m", "--machine", type=str, default=None, help="Machine name"
)


def echo_success(message: str = "Success!"):
    click.secho(message, fg="green")


@util.command()
@click.argument("path", type=str, required=False, default=".")
@click.option("-a", "--hidden", is_flag=True, help="Show hidden files")
@click.option("-l", "--long", is_flag=True, help="Long format")
@click.option("-R", "--recursive", is_flag=True, help="Recurse directories")
@click.option("--delimiter", type=str, default="/", help="Delimiter recursive joining")
@click.option(
    "--max-calls",
    type=int,
    default=100,
    help="Maximum API calls allowed during recursion",
)
@MACHINE_OPTION
@pass_config
def ls(
    config: LazyConfig,
    machine: str | None,
    path: str,
    hidden: bool,
    long: bool,
    delimiter: str,
    recursive: bool,
    max_calls: int,
):
    """GET `/utilities/ls`"""
    machine = machine or config.default_machine
    if not recursive:
        max_calls = 1
        raise_on_max = False
    else:
        raise_on_max = True
    for result in config.client.ls_recurse(
        machine,
        path,
        show_hidden=hidden,
        delimiter=delimiter,
        max_calls=max_calls,
        raise_on_max=raise_on_max,
    ):
        if long:
            print(f"- {json.dumps(result)}")
        else:
            print(
                "  " * result.get("depth", 0)
                + result["name"]
                + ("/" if result.get("type") == "d" else "")
            )


@util.command("file")
@click.argument("path", type=str)
@MACHINE_OPTION
@pass_config
def file_type(config: LazyConfig, machine: str | None, path: str):
    """GET `/utilities/download`"""
    print(config.client.file_type(machine or config.default_machine, path))


@util.command()
@click.argument("path", type=str)
@click.option("-d", "--dereference", is_flag=True, help="Dereference symlinks")
@MACHINE_OPTION
@pass_config
def stat(config: LazyConfig, machine: str | None, path: str, dereference: bool):
    """GET `/utilities/stat`"""
    print(config.client.stat(machine or config.default_machine, path, dereference))


@util.command()
@click.argument("path", type=str)
@MACHINE_OPTION
@pass_config
def checksum(config: LazyConfig, machine: str | None, path: str):
    """GET `/utilities/checksum`"""
    print(config.client.checksum(machine or config.default_machine, path))


@util.command()
@click.argument("path", type=str)
@MACHINE_OPTION
@pass_config
def view(config: LazyConfig, machine: str | None, path: str):
    """GET `/utilities/view`"""
    print(config.client.view(machine or config.default_machine, path))


@util.command()
@click.argument("path", type=str)
@MACHINE_OPTION
@pass_config
def download(config: LazyConfig, machine: str | None, path: str):
    """GET `/utilities/download`"""
    config.client.simple_download(machine or config.default_machine, path)
    echo_success()


@util.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False))
@click.argument("target_dir", type=str, default=".", required=False)
@click.argument("target_name", type=str, required=False)
@MACHINE_OPTION
@pass_config
def upload(
    config: LazyConfig,
    machine: str | None,
    source: str,
    target_dir: str,
    target_name: str | None,
):
    """GET `/utilities/upload`"""
    config.client.simple_upload(
        machine or config.default_machine, source, target_dir, target_name
    )
    echo_success()


@util.command()
@click.argument("path", type=str)
@click.option("-p", "--parents", is_flag=True, help="Create parent directories")
@MACHINE_OPTION
@pass_config
def mkdir(config: LazyConfig, machine: str | None, path: str, parents: bool):
    """GET `/utilities/mkdir`"""
    config.client.mkdir(machine or config.default_machine, path, parents)
    echo_success()


@util.command()
@click.argument("path", type=str)
@MACHINE_OPTION
@pass_config
def rm(config: LazyConfig, machine: str | None, path: str):
    """GET `/utilities/rm`"""
    config.client.simple_delete(machine or config.default_machine, path)
    echo_success()


@main.group()
def compute():
    """`/compute/` endpoints."""


@compute.command()
@click.option("-j", "--jobs", type=str, help="Comma delimited list of job IDs")
@click.option("-s", "--start", type=str, help="Start date/time")
@click.option("-e", "--end", type=str, help="End date/time")
@MACHINE_OPTION
@pass_config
def acct(
    config: LazyConfig,
    machine: str | None,
    jobs: str | None,
    start: datetime | None,
    end: datetime | None,
):
    """GET `/compute/acct`"""
    jobs = jobs.split(",") if jobs else None
    print(config.client.poll(machine or config.default_machine, jobs, start, end))


@compute.command()
@click.option("-j", "--jobs", type=str, help="Comma delimited list of job IDs")
@MACHINE_OPTION
@pass_config
def jobs(config: LazyConfig, machine: str | None, jobs: str | None):
    """GET `/compute/jobs`"""
    jobs = jobs.split(",") if jobs else None
    print(config.client.poll_active(machine or config.default_machine, jobs))


@compute.command()
@click.argument("job", type=str)
@MACHINE_OPTION
@pass_config
def cancel(config: LazyConfig, machine: str | None, job: str):
    """DELETE `/compute/job/{jobid}`"""
    print(config.client.cancel(machine or config.default_machine, job))


@compute.command()
@click.argument("script_path", type=str)
@click.option("-l", "--local", is_flag=True, help="Whether the path is local")
@MACHINE_OPTION
@pass_config
def submit(config: LazyConfig, machine: str | None, script_path: str, local: bool):
    """POST `/compute/jobs/upload` or POST `/compute/jobs/path`"""
    print(config.client.submit(machine or config.default_machine, script_path, local))


if __name__ == "__main__":
    main(help_option_names=["-h", "--help"])
