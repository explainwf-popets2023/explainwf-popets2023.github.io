Docker Initialization
---------------------

1. Build the Docker image:

  `$ docker build -t IMAGE_NAME`

2. Start up a Docker container for the experiments and mount the necessary code
and data directories:

  ```
  $ docker run --name CONTAINER_NAME -it \
  --mount type=bind,source=CODE_DIR,target=/mnt/bind/code \
  --mount type=bind,source=DATA_DIR,target=/mnt/bind/data \
  IMAGE_NAME /bin/bash
  ```

3. Run `$ /mnt/bind/code/download.bash` and `$ /mnt/bind/code/patch.bash`
to download the necessary external scripts.

Data Preprocessing
------------------

  ** A convenience script performing all the steps below is provided for
  convenience. Running

  ```
  $ source docker_filepaths.bash && ./preprocess_data.bash
  ```

  will create all the required directories and files in the `./out/` directory.

1. Move into the code directory: `$ cd /mnt/bind/code`

2. Create a directory for intermediate files: `$ mkdir out`

3. Run the `extract.py` script to clean and format the raw data:

  For the closed-world experiments in Section 3.3, the trace files are:
    - `/mnt/bind/data/wget2.torcat.log`
    - `/mnt/bind/data/wget2-traces-tornet-net_0.25-load_2.0.log`

  The command to run for these files is:

  ```
  $ python3 extract.py -n 200 -r -c <TRACES_FILEPATH> /mnt/bind/data/urls.txt \
  ./out/<CLEANED_FILE>.pkl
  ```

  For the rest of the experiments in Sections 4 and 5, the trace files follow
  the pattern

    `/mnt/bind/data/wget2-traces-tornet-net_0.25-load_<LOAD>-<SEED>.log`

  where load is in the set {1.5, 2.0, 2.5} and seed is in the set {a, b, c}.

  The command to run for these files is:

  ```
  $ python3 extract.py -f -n 300 --nkeep_port_80 30000 -c <TRACES_FILEPATH> \
  /mnt/bind/data/urls.txt ./out/<CLEANED_TRACES>.pkl
  ```

4. Create a label file for each type of experiment to run. A single label file
keeps the labels consistent across data sets.

  To create a label set for closed-world multiclass experiments:

  ```
  $ python3 make_labels.py out/closed_labels.pkl <CLEANED_TRACES>...
  ```

  For example:

  ```
  $ python3 make_labels.py out/closed_labels.pkl out/wget2-cleaned-real.pkl \
  out/wget2-cleaned-shadow.pkl 
  ```

  where 

    - `./out/wget2-cleaned-real.pkl` was produced by calling
      `extract.py` on `/mnt/bind/data/wget2.torcat.log`
    - `./out/wget2-cleaned-shadow.pkl` was produced by calling
      `extract.py` on `/mnt/bind/data/wget2-traces-tornet-net_0.25-load_2.0.log`

  To create a label set for open-world binary experiments:

  ```
  $ python3 make_labels.py -n 5 out/open_labels.pkl <CLEANED_TRACES>...
  ```

5. Split each cleaned trace file into training and testing tests:

  For closed-world multiclass experiments:

  ```
  $ python3 split_train_test.py <CLEANED_TRACES> <LABELS> <OUTPUT_DATASET>
  ```

  For example:

  ```
  $ python3 split_train_test.py ./out/wget2-cleaned-real.pkl \
  ./out/closed_labels.pkl ./out/wget2-real-dataset.pkl
  ```

  For open-world binary experiments:

  ```
  $ python3 split_train_test.py -b <CLEANED_TRACES> <LABELS> <OUTPUT_DATASET>
  ```

Model Training
--------------

1. To train a model on the closed world datasets:

  ```
  $ python3 ./train_models.py <DATASET_FILE> <MODEL_DIRECTORY>
  ```

  For example

  ```
  $ python3 ./train_models.py ./out/split/section3/wget2.torcat.pkl \
  ./out/models
  ```

  will train models on the Tor wget2 dataset. The models will be output to the
  specified model directory. The `-t <TAG>` flag can be passed to the script
  to provide a tag that will be included in the output model names.

2. Training a model on the open-world datasets is similar, but the `-o` should
be provided:

  ```
  $ python3 ./train_models.py -o 5 <DATASET_FILE> <MODEL_DIRECTORY>
  ```

Model Testing
-------------

1. To test a model on a dataset:

  ```
  $ python3 test_models.py <DATASET_FILE> <OUTPUT_STATS_FILE>
    <MODEL_FILEPATH>...
  ```

  For example

  ```
  $ python3 test_models.py ./out/split/section3/wget2.torcat.pkl \
    output.json ./out/models/kfp.tar.gz
  ```

  Multiple models can be specified to run on the input dataset. Models cannot be
  mixed across dataset types: a closed-world model must be tested on a
  closed-world dataset and an open-world model must be tested on an open-world
  dataset.