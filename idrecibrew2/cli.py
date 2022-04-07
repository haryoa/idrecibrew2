import fire
from .experiment import Experiment

class CLI:
    
    def run_sc(self, scenario, gpus, is_test: bool = False):
        experiment = Experiment(scenario=scenario, gpus=gpus, is_test=is_test)
        experiment.run()
        

if __name__ == '__main__':
    fire.Fire(CLI)
