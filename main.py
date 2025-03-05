import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
# @hydra.main(config_name='config'):Hydra automatically loads the configuration file named config.yaml
##The configuration is passed as a DictConfig object to the go function.
#This configuration file (config.yaml) likely contains parameters like project settings, data paths, and model hyperparameters.

@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    ## The names will be taken from the config file
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    # Hydra can change the working directory when running the script,
    # so the original root path needs to be saved before anything else gets executed
    root_path = hydra.utils.get_original_cwd()




    # Check which steps we need to execute
    # this is a trick to help in de-bugging, in case there is a problem in any one of the components
    # it will help us to run that specific compoennt
    # By default, the script will run everything

    # execute_steps is a parameter and a user can pass values to the parameter from the command line.
    # it will take a value like 
        #   comma sepaarted objects - "download,preprocess,evaluate"  - To be passed in this format during de-bugging
        #   List specified in the config file -  ["download", "preprocess", "evaluate"]     To be passed in this format during actual execution
    # that value is saved in a temp variable called steps_to_execute 

    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        assert isinstance(config["main"]["execute_steps"], list)
        steps_to_execute = config["main"]["execute_steps"]

    # Download step
    if "download" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "download"),
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            },
        )

    if "preprocess" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),
            "main",
            parameetrs = {
                "input_artifact": "raw_data.parquet:latest" ,
                "artifact_name": "preprocessed_data.csv" ,
                "artifact_type": "preprocessed_data",
                "artifact_description": "data with pre processing applied"
            },
        )

        

    if "check_data" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            "main",
            parameters ={
                "reference_artifact": config["data"]["reference_dataset"]
                "sample_artifact" : "preprocessed_data.csv:latest"
                "ks_alpha" : config["data"]["ks_alpha"]
            }, 
        )

        
        

    if "segregate" in steps_to_execute:

       _ = mlflow.run(
        os.path.join(root_path, "segregate"),
        "main",
        parameters = {
            "input_artifact" : "preprocessed_data.csv:latest",
            "artifact_root": "data",
            "artifact_type": "segregated_data",
            "test_size": config["data"]["test_size"],
            "stratify": config["data"]["stratify"]
        },

       )

    # this training is a one time job. The idea is to do it once, get all the model specs and save it in the model registry
    # the evaluate step will take it from the model registry as the meta data and then use it to evaluate the model
    if "random_forest" in steps_to_execute:

        # Serialize decision tree configuration
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        _ = mlflow.run (
            os.path.join(root_path, "random_forest"),
            "main",
            parameters = {
                "train_data": "data_train.csv:latest",
                "model_config": model_config,
                # a name for teh export artifact. This will contain our inference pipeline in wandb
                "export_artifact": config["random_forest_pipeline"]["export_artifact"],
                "random_seed": config["main"]["random_seed"],
                "val_size": config["data"]["test_size"],
                "stratify":config["data"]["stratify"]
            }
        )

    if "evaluate" in steps_to_execute:

       _ = mlflow.run(
        os.path.join(root_path, "evaluate"),
        "main",
        parameters = {
            # the model export comes from the previous component. Note that it follows the same naming convention as specified in the config file
            "model_export": f"{config['random_forest_pipeline']['export_artifact']}:latest",
            "test_data":"data_test.csv:latest"
        }

       )


if __name__ == "__main__":
    go()
