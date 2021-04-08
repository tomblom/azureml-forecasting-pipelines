from azureml.core.workspace import Workspace


def main():
    ws = Workspace.from_config()
    print(ws.name)


if __name__ == "__main__":
    main()
