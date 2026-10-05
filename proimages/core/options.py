from pydantic import BaseModel, ConfigDict


class ProcessOptions(BaseModel):
    """Which optional pipeline stages to run, and how.

    Each stage adds its own sub-options field when it is implemented, and setting that field
    turns the stage on, so no option can ask for a stage that does not exist yet. Unknown
    fields are rejected rather than ignored.

    Holds no file paths: the API accepts this object as JSON from any client, and a path in
    it would let a request make the server read or write files.
    """

    model_config = ConfigDict(extra="forbid")
