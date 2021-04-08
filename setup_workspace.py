import os
from dotenv import load_dotenv
from azureml.core import Workspace


def main():
    load_dotenv()
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace_name = os.getenv("WORKSPACE_NAME")

    try:
        ws = Workspace(subscription_id=subscription_id,
                       resource_group=resource_group,
                       workspace_name=workspace_name)
        print("Workspace configuration succeeded. Skip the workspace creation steps below")
    except:
        print("Workspace does not exist. Please create your AzureML Workspace before continuing")
        raise SystemExit

    ws.write_config()
    print("Workspace configuration succeeded.")


if __name__ == "__main__":
    main()
