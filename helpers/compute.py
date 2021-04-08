import os
from dotenv import load_dotenv
from azureml.core import Workspace
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.exceptions import ComputeTargetException


def get_or_create_compute(ws: Workspace):
    load_dotenv()
    compute_name = os.getenv("AML_COMPUTE_CLUSTER_NAME")
    vm_size = os.getenv("AML_COMPUTE_CLUSTER_CPU_SKU")
    max_nodes = int(os.getenv("AML_CLUSTER_MAX_NODES"))
    min_nodes = int(os.getenv("AML_CLUSTER_MIN_NODES"))
    priority = os.getenv("AML_CLUSTER_PRIORITY")

    try:
        compute_target = ComputeTarget(workspace=ws, name=compute_name)
        print("Found existing cluster, use it.")
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                               min_nodes=min_nodes,
                                                               max_nodes=max_nodes,
                                                               vm_priority=priority)
        compute_target = ComputeTarget.create(ws, compute_name, compute_config)

    compute_target.wait_for_completion(show_output=True)
    return compute_target
