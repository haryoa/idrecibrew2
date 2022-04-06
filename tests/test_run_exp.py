from idrecibrew2.experiment import Experiment


def test_run_full_exp():
    experiment = Experiment(scenario='indobert-v2', gpus=[0], is_test=True)
    experiment.run()
