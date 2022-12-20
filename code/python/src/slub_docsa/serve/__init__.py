"""REST service.

This package provides a REST service based on an OpenAPI definition via [swagger](https://swagger.io/). A full
documentation of the REST endpoints (swagger user interface) can be found after starting the REST service at the URL
`http://<host>:<port>/v1/ui/`. The OpenAPI definition yaml file can be found
[here](https://github.com/slub/docsa/blob/master/code/python/src/slub_docsa/openapi.yaml).

Note that:

- Currently, there is no support for user authentication, quotas, request limiting, or protection against DoS attacks.
  The service should **only be used in a secured environment** (behind a proxy, local network, trustworthy clients,
  etc.)

## Available REST endpoints

Language Detection
- `/languages` - list all supported languages, currently only "de" and "en"
- `/languages/detect` - allows to determine the language of a document via automatic language detection

Model Discovery and Classification
- `/models` - list all available classification models
- `/models/{model_id}` - provides detail information about a model (e.g. version, creation date)
- `/models/{model_id}/classify` - classifies one or multiple documents (optimized for performance)
- `/models/{model_id}/classify_and_describe` - classifies documents and provides detail information about the
  predicted subjects
- `/models/{model_id}/subjects` - lists all subjects supported by a model

Classification Schema Information
- `/schemas` - lists all subject schema (`rvk`, `ddc`, `bk`)
- `/schemas/{schema_id}/subjects` - list (root) subjects of a schema
- `/schemas/{schema_id}/subjects/info` - provides detail information about a subject (labels, ancestors, children)

## Starting the REST Service

The REST service can be started via the command line interface and the command `slub_docsa serve`. The following
optional parameters can be provided:

```
usage: slub_docsa serve [-h] [-v] [-s]
                        [--data_dir DATA_DIR] [--resources_dir RESOURCES_DIR]
                        [--cache_dir CACHE_DIR] [--figures_dir FIGURES_DIR]
                        [--serve_dir SERVE_DIR]
                        [--all]
                        [--debug]
                        [--host HOST]
                        [--port PORT]
                        [--threads THREADS]
```

Most notably:

- `--serve_dir`<br /> Allows to define the directory from where classification models are loaded. Providing the
  directory `/path/to/serve` will scan for classification models at `/path/to/serve/classification_models`. By default,
  the directory `<data_dir>/runtime/serve` is used.
- `--all`<br /> Instructs the REST service to pre-load all available classification models. This might fail if there is
  not enough memory available to load all models. By default (if `--all` is not provided), the REST service will only
  load one classification model at the same time. If a request requires a different model (than the currently loaded
  one), the previous model will be unloaded, and the requested model will be loaded. Depending on the model, loading it
  will take some time and, therefore, the response time of requests to the REST service could be impacted.
- `--debug`<br />
  Starts the REST service in debug mode, which will print additional log messages in case of problems and automatically
  reloads the REST service in case of source file changes (for development only).
- `--host`<br />
  Allows to specify the host address that is used when listening for requests (default 0.0.0.0)
- `--port`<br />
  Allows to specify the port that is used when listening for requests (default 5000)
- `--threads`<br />
  Specifies the number of application threads that are used process requests (only in production mode, default 4)


## Generate Models for the REST Service

Classification models are loaded from the directory `<serve_dir>/classification_models/<model_directory>`. These
persisted models can be generated either via the command line interface and the command
`slub_docsa classify [dataset] train` or by calling the Python API method
`slub_docsa.data.store.model.save_as_published_classification_model` with an already trained model and some meta
information.

### Via CLI

When training a classification model via the command line interface, it is stored automatically after the training is
finished. The stored model can be found at `<persist_dir>`, see `slub_docsa.cli`, by default
`<data_dir>/runtime/cache/models/<dataset_variant>__<model_type>`.

In order for the model to be discovered by the REST service, it needs to be copied to the corresponding directory
`<serve_dir>/classification_models/`.

Since some meta information is not available when issuing the training command, you need to specify additional
information about the model manually by editing the generated `classification_model_info.json` file that stores all
information about a model. The following information need to be completed with suitable information:

```json
{
    "model_id": "some-unique-identifier-for-the-model",
    "model_type": "the-model-type-name",
    "model_version": "0.0.0",
    "schema_id": "rvk",
    "creation_date": "2022-12-19 13:03:37",
    "supported_languages": ["de"],
    "description": "some description",
    "tags": [],
    "slub_docsa_version": "0.1.0.dev1-42f3661",
    "statistics": {
        "number_of_training_samples": 99955,
        "number_of_test_samples": 0,
        "scores": {}
    }
}
```

Basic documentation for these properties can be found at `slub_docsa.serve.common.PublishedClassificationModelInfo`.

When evaluating models via the `slub_docsa experiments [dataset] classify_many` command, the calculated score values
are already included in the model information json file.

### Via Python API

Classification models can also be stored via the Python method
`slub_docsa.data.store.model.save_as_published_classification_model`. A model implementation needs to be a subclass of
the `slub_docsa.common.model.PersistableClassificationModel` in order to be able to be stored. In addition to that, the
same meta information describe above (see `slub_docsa.serve.common.PublishedClassificationModelInfo`) need to be
provided.

For example, lets save a simple k-nearest neighbor classifier:

```python
model = ScikitClassifier(
    predictor=KNeighborsClassifier(n_neighbors=1, metric="cosine"),
    vectorizer=ScikitTfidfVectorizer(max_features=10000),
)

# do training
model.fit(train_documents, train_incidences)

# save model
save_as_published_classification_model(
    directory="/path/to/the/model/directory",
    model=model,
    subject_order=subject_order,
    model_info=PublishedClassificationModelInfo(
        model_id="some-unique-identifier-for-a-model",
        model_type="tfidf_10k_knn_k=1",
        model_version="0.0.0",
        schema_id="rvk",
        creation_date="2022-12-19 13:03:37",
        supported_languages=["de"],
        description="short description",
        tags=[],
        slub_docsa_version=slub_docsa.__version__,
        statistics=PublishedClassificationModelStatistics(
            number_of_training_samples=len(train_documents),
            number_of_test_samples=0,
            scores={}
        )
    )
)
```

Note that the `model_type` needs to be known to the REST service implementation. By default, only a fixed set of
classification models are supported, see
`slub_docsa.serve.models.classification.common.get_all_classification_model_types`
(which can be extended as described below).

## Customizing the REST service

The default REST service started via the command line only supports a fixed set classification model implementations,
a fixed set of subject schema (RVK, DDC, BK), a fixed language detection strategy, and two basic model loading
strategies (all vs. one-at-a-time).

However, the REST service can be extended by providing a custom implementation of the REST service backend API, which
is documented at `slub_docsa.serve.common`. It consists of an abstract REST service base class that will be called for
each respective REST endpoint. When starting the REST service via `slub_docsa.serve.app.create_webapp`, a custom REST
service implementation can be provided.

For example, a custom classification model generator can be added like this:

```python
model_types = get_all_classification_model_types()
model_types.update({
    "my-custom-model": lambda subject_hierarchy, subject_order: MyCustomModelImplementation(),
})
```

Then, the REST service can be started with:

```python
schema_generators = default_schema_generators()
schema_rest_service = SimpleSchemaRestService(schema_generators)
model_rest_service = SingleStoredModelRestService(models_dir, model_types, schema_rest_service, schema_generators)
lang_rest_service = LangidLanguagesRestService()

create_webapp(SimpleRestService(model_rest_service, schema_rest_service, lang_rest_service)).run(host=HOST, port=PORT)
```
"""
