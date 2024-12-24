import os

import numpy as np
import wandb


class Visualizer(object):
    def __init__(self, opt):
        self.opt = opt
        if self.opt.wandb_log:
            wandb.login()
            self.run = wandb.init(
                project=opt.project_name,
                entity=opt.entity_name,
                name=opt.run_name,
                # Track hyperparameters and run metadata
                config=opt,
            )
            self.pos = np.arange(0, 201)
        wandb_path = f"{opt.logs_dir}/{opt.run_name}"
        if not os.path.exists(wandb_path):
            os.makedirs(wandb_path)

    def print_current_errors(self, epoch, i, errors, t):
        message = "(epoch: %d, iters: %d, time: %.3f) " % (epoch, i, t)
        for k, v in errors.items():
            message += "%s: %.3f " % (k, v)
        print(message)

    def plot_current_errors(self, errors, step):
        if not self.opt.wandb_log:
            return

        wandb.log(errors, step=step)

    def display_current_results(self, predicts, targets, meta, step):
        if not self.opt.wandb_log:
            return

        for i in range(predicts.shape[0]):
            wandb.log(
                {
                    "Example"
                    + str(i): wandb.plot.line_series(
                        xs=[self.pos, self.pos],
                        ys=[np.array(predicts[i].detach().cpu()), np.array(targets[i].detach().cpu())],
                        keys=["predict index", "target index"],
                        xname="pos",
                    )
                }
            )

    def save_model_wandb(self, logs_dir, run_name):

        if not self.opt.wandb_log:
            return

        artifact = wandb.Artifact(run_name + "best_corr_model", type="model")
        save_path = os.path.join(logs_dir, run_name, "checkpoint.pth")
        artifact.add_file(save_path)
        self.run.log_artifact(artifact)
