# API catalogs

Each JSONC file describes one simulated service. The endpoint descriptions and
shapes were independently derived from the public API documentation identified
by the provenance notice at the top of every catalog; AutomationBench does not
call those live APIs.

The root contains only the service's `api` name and its endpoint definitions.
Transport URLs and standalone response-model catalogs are not part of the
endpoint-ID interface.

An endpoint's `request` is a complete JSON Schema Draft 2020-12 object schema.
It is the source of truth for both model-facing discovery and runtime request
validation. A missing `request` means that the endpoint accepts no body. Some
public documentation does not identify a field's type; those fields use an
unconstrained schema, so the catalog does not guess or narrow the accepted input.

The optional endpoint metadata below describes simulator routing that is not
part of the public request:

- `implemented`: excludes a documented endpoint when no simulator handler exists.
- `internalPath`: supplies path fields owned by the simulated workspace.
- `hiddenParams`: removes adapter-only parameters from model-facing discovery.
- `fixedParams` and `fixedBody`: add constants needed by a shared handler.
- `forbiddenBody`: rejects fields that select a different operation on a shared route.
- `bodyAliases`: renames a public request field for the simulator adapter.

Keeping these decisions beside the endpoint makes the runtime contract a small
loader and executor instead of a second, handwritten interpretation of the
catalog prose.
