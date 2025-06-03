import os
import pickle
import shutil

import pandas as pd


class Logger:
    def __init__(
        self,
        logging_path: str,
        hpo_keys: dict,
    ) -> None:
        self.hpo_keys = hpo_keys
        self.logging_path = os.path.join(logging_path, "logs")
        os.makedirs(self.logging_path, exist_ok=True)

    def read_logs(self):
        """It enables to relauch an experiment after a system failure"""

        log_files = [file for file in os.listdir(self.logging_path) if file.startswith("round_") and file.endswith("_logs.csv")]

        if len(log_files) == 0:
            # The first round didn't complete, so we launch xp from start
            if os.path.exists(os.path.join(self.logging_path, "round_0")):
                shutil.rmtree(os.path.join(self.logging_path, "round_0"))
            return False, None, None, None

        round_index = max(
            map(lambda x: int(x.split("_")[1]), log_files),
        )
        log_df = pd.read_csv(
            os.path.join(self.logging_path, f"round_{round_index}_logs.csv"),
        )

        agent_files = os.listdir(
            os.path.join(self.logging_path, f"round_{round_index}"),
        )
        agent_params = {}
        for file in agent_files:
            global_index = int((file.split("_")[1]).split(".")[0])
            file_path = os.path.join(
                self.logging_path,
                f"round_{round_index}",
                file,
            )
            params = pickle.load(open(file_path, "rb"))  # noqa: SIM115
            agent_params.update({global_index: params})

        return True, round_index, log_df, agent_params

    def write_logs(self, agents, round_index: int):
        agent_keys = [key for key in vars(agents[0]) if "param" not in key]
        log_keys = agent_keys + self.hpo_keys
        log_df = {key: [] for key in log_keys}

        for agent in agents:
            for key in agent_keys:
                log_df[key].append(agent.__getattribute__(key))
            for hp_name in self.hpo_keys:
                log_df[hp_name].append(agent.hyperparameters[hp_name])

        log_df = pd.DataFrame(log_df)
        log_df.to_csv(
            os.path.join(self.logging_path, f"round_{round_index}_logs.csv"),
            index=False,
        )

        # Save all agent parameters
        self._save_params(agents, round_index)

        # Save the best model alongside the CSV log
        best_agent_index = int(log_df.iloc[log_df["reward"].argmax()].loc["index"])
        self._save_best_model(agents[best_agent_index], round_index)

        # Clean up previous round to save disk space
        if round_index >= 1:
            self._remove_previous_round(round_index)

    def _save_params(self, agents, round_index):
        for global_index, agent in enumerate(agents):
            save_path = os.path.join(self.logging_path, f"round_{round_index}")
            os.makedirs(save_path, exist_ok=True)
            pickle.dump(
                agent.params,
                open(os.path.join(save_path, f"agent_{global_index}.pkl"), "wb"),  # noqa: SIM115
            )

    def _save_best_model(self, best_agent, round_index: int):
        """Save the best model parameters alongside the CSV logs"""
        best_model_path = os.path.join(self.logging_path, f"round_{round_index}_best_model.pkl")
        pickle.dump(best_agent.params, open(best_model_path, "wb"))  # noqa: SIM115

    def _remove_previous_round(self, round_index: int):
        """Trick to gain some disk space, keep only the models needed to reload xp"""
        try:
            shutil.rmtree(
                os.path.join(self.logging_path, f"round_{round_index - 1}"),
            )
        except FileNotFoundError:
            return None
