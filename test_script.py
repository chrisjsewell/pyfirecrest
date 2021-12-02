"""
Required:

Scheduler commands:

_get_detailed_job_info_command
    sacct --format=<AllocCPUS,Account,...> --parsable --jobs=<job_id>

_get_submit_command
    sbatch <script path>

_get_joblist_command
    SLURM_TIME_FORMAT='standard' squeue --noheader -o <%i,...> -u <user> --jobs=<1,...>

_get_kill_command
    scancel <job_id>

"""

import json
import os

import firecrest as f7t


def custom_decorator(func):
    def _decorator(self, *args, **kwargs):
        return self.keycloak.account_login(func)(self, *args, **kwargs)
    return _decorator

class MyAuthorizationClass:
    def __init__(self, client_id, client_secret):
        self.keycloak = f7t.ClientCredentialsAuthorization(
            client_id,
            client_secret,
            "https://auth.cscs.ch/auth/realms/cscs/protocol/openid-connect/token"
        )

    @custom_decorator
    def get_access_token(self):
        return self.keycloak.get_access_token()

class Transport:
    def __init__(self, client: f7t.Firecrest, machine_name: str):
        self._client = client
        self._machine_name = machine_name

    @property
    def client(self):
        return self._client

    @property
    def machine_name(self):
        return self._machine_name

    # def chdir(self, path):
    #     """Change directory to 'path'"""

    def chmod(self, path: str, mode: int):
        """Change permissions of a path."""
        self.client.chmod(self.machine_name, path, mode)

    def chown(self, path: str, uid: int, gid: int):
        """Change owner and group id of a path."""
        self.client.chown(self.machine_name, path, uid, gid)

    def listdir(self, path='.', pattern=None):
        """List directory contents."""
        # TODO pattern
        files = self.client.list_files(self.machine_name, path)
        return [f['name'] for f in files]
    


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), ".firecrest-secrets.json")) as f:
        secrets = json.load(f)
    

    # Setup the client with the appropriate URL and the authorization class
    client = f7t.Firecrest(firecrest_url="https://firecrest.cscs.ch", authorization=MyAuthorizationClass(
        client_id=secrets["clientID"],
        client_secret=secrets["clientSecret"])
    )

    transport = Transport(client, "daint")
    print(transport.listdir("/scratch/snx3000/csewell/"))

    # systems = client.all_systems()
    # print("*** ALL SYSTEMS:")
    # print(systems)

    # services = client.all_services()
    # print("*** ALL SERVICES:")
    # print(services)

    # print("*** LISTING FILES ON SCRATCH")
    # #files = client.list_files("daint", "/users/csewell/") # NOT ALLOWED 400
    # files = client.list_files("daint", "/scratch/snx3000/csewell")
    # print(files)

