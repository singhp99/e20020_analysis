from datetime import datetime


class Event:
    """
    Represents an event that can be published to a topic within the event processing engine.
    Each event carries a topic, partition, value, and timestamp indicating when it was created.

    Attributes:
        sender (int): The ID of the sender, initialized to 0 and can be updated to reflect the sender's ID.
        timestamp (datetime): The time when the event was created. Defaults to the current time if not specified.
        topic (str): The topic to which the event is related.
        partition (int): The partition within the topic that the event belongs to.
        value (any): The value or content of the event.

    Methods:
        __repr__() -> str: Returns a string representation of the event, including its timestamp, topic, partition, and value.
    """

    def __init__(
        self,
        topic: str,
        partition: str,
        value: any,
        timestamp: datetime = datetime.now(),
    ) -> None:
        self.sender = 0  # type: int
        self.timestamp = timestamp  # type: datetime
        self.topic = topic  # type: str
        self.partition = partition  # type: str
        self.value = value  # type: any

    def __repr__(self) -> str:
        return f"{self.timestamp} - {self.topic}/{self.partition}: {self.value}"
