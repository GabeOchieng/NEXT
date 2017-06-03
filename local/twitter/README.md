# Running NEXT locally
This document contains instructions on how to run a NEXT experiment on your local machine.


## Dependencies

To start the NEXT backend, you need a machine with the following things installed:

```
docker
docker-compose
python2.7
```

`docker` can be installed via the [Docker install guide]. `docker-compose` can
be installed via `pip install docker-compose`.

Optionally, you need the PyPI packages `requests` and `pyyaml` to run the `launch.py` script in this directory:

`pip install requests yaml`


### Using MacOS
If using MacOS/OS X, download [Docker for Mac], not [Docker Toolbox] It
provides an easier interface to get started.

[Docker for Mac]:https://docs.docker.com/engine/installation/mac/#/docker-for-mac

[Docker install guide]:https://docs.docker.com/v1.8/installation/

[Docker Toolbox]:https://www.docker.com/products/docker-toolbox


## Starting the backend

First clone this repository and navigate to the `NEXT/local/` directory.  

To start up the NEXT backend, run `./docker_up.sh [host]` where `host`
is the IP or hostname of the machine running the backend.  You may
optionally provide a path to the repo if you are running the
`docker_up` script from a different directory.  For example:

```
./docker_up.sh [host] [/path/to/NEXT]
```

The default will assume host is `localhost` and `NEXT` is located at `../../`:

```
./docker_up.sh
```

The first time you run this, docker will download and build the images, which will take a few minutes.

Once the backend is launched, you should be able to go to `http://localhost:8000/home` to see the NEXT homepage.


## Starting an experiment

Once the backend is running you can launch an experiment.  To set up an experiment, you need to create a yaml file specifying all
the required parameters and data.  Usually you'll want to point to an external .json file for the data.
See the example `NEXT/local/twitter/init.yaml`.

You can launch an experiment by clicking on the *Experiment launch* link on the NEXT homepage and uploading the
appropriate `init.yaml` file (no need to specify a targets file).
Alternatively, you can launch the experiment by running:
```bash
python launch.py twitter/init.yaml
```


## Configuring experiments

See https://github.com/dconathan/NEXT/blob/master/apps/MulticlassClassification/myApp.yaml for details on all the configuration options.
But most importantly, you have:

- cache_size: this is how many queries to queue up each iteration of the algorithm. Smaller is better but not too small. Aim for 5-10 depending on how many people you expect to be labeling at the same time.
- label_mode: 'onehot' means each target can be one and only one category. 'multilabel' means a target can be any number of categories ('check all that apply')
- classes: these define what labels to assign. Note that the labels will be binary lists that correspond to this parameter. E.g. if class=['yes', 'no'] then a label of [1, 0] corresponds to yes and [0, 1] is no.
- num_tries: how many things people label before they get the popup and have to hit refresh to label more
- test_size: this is how many queries that get randomly sampled to be put in the holdout/validation set. You need a validation set to evaluate performance and tune thresholds. You probably want this around 100-300, but maybe more if you have lots of categories.

## Getting the data

To download the labels once the experiment is done, go to the experiment dashboard page and click the "Experiment data" link.  This will link to a .json file that has the important data related to the experiment.

In it you will find the keys `train_labels` and `test_labels`, which are themselves key-value stores of the form `{row_of_target_file: label}`
`train_labels` contain the actively sampled targets that should be used for training, `test_labels` contain randomly sampled targets to be used for validation.

