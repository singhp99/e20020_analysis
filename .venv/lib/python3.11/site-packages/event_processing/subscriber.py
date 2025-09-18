from event_processing.event import Event


class Subscriber:
    """
    A base class for subscribers in an event processing system. Subscribers can send and receive events.

    Attributes:
        id (int): Unique identifier for the subscriber.

    Methods:
        send(event: Event): Sends an event to the event processing engine.
        receive(event: Event): Handles receiving an event from the event processing engine. Should be implemented by subclasses.
    """

    _id = 0  # Class-level attribute to track the last assigned subscriber ID

    @staticmethod
    def get_id() -> int:
        """
        Generates a unique ID for a new subscriber instance.

        Returns:
            int: A unique subscriber ID.
        """
        Subscriber._id += 1
        return Subscriber._id

    def __new__(cls, *args, **kwargs) -> "Subscriber":
        """
        Overrides the default __new__ method to assign a unique ID to each subscriber instance.

        Returns:
            Subscriber: A new instance of Subscriber with a unique ID.
        """
        instance = super(Subscriber, cls).__new__(cls)
        instance.id = Subscriber.get_id()
        return instance

    def _send(self, event: Event) -> None:
        """
        Callback method used by the engine to send an event. Intended to be linked to the engine's send mechanism.

        Parameters:
            event (Event): The event to send.
        """
        pass

    def send(self, event: Event) -> None:
        """
        Public method to send an event, setting the sender to the current subscriber's ID.

        Parameters:
            event (Event): The event to be sent.
        """
        event.sender = self.id
        self._send(event)

    def receive(self, event: Event) -> None:
        """
        Method to be implemented by subclasses for handling received events.

        Parameters:
            event (Event): The event received from the event processing engine.
        """
        pass
