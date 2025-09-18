from threading import Thread
from queue import Queue


class Engine:
    """
    A simple event processing engine that allows objects to subscribe to topics and receive events.
    Events are queued and processed in a single-threaded, deterministic manner to ensure the order of events is preserved.

    Attributes:
        subscriptions (dict): A dictionary mapping topics to subscribers and their callback functions.
        events (Queue): A queue for storing events before they are processed.
        lock (bool): A flag to prevent concurrent execution of the consume method.

    Methods:
        subscribe(subscriber, topic): Subscribes a subscriber object to a specified topic.
        unsubscribe(subscriber_id, topic): Unsubscribes a subscriber from a specified topic.
        inject(event): Adds an event to the events queue and triggers its processing.
        consume(): Processes all events in the queue, ensuring each subscriber receives events for their subscribed topics.
    """

    def __init__(self):
        self.subscriptions = {}
        self.events = Queue()
        self.lock = False

    def subscribe(self, subscriber, topic):
        """
        Registers a subscriber to a specific topic. Each subscriber is identified by a unique ID,
        and their receive method is associated with the topic for event notifications.

        Parameters:
            subscriber (object): The subscriber object that wants to receive events.
                                 Must have an 'id' attribute and a 'receive' method.
            topic (str): The topic to which the subscriber wants to subscribe.
        """

        # If it is a new topic
        if topic not in self.subscriptions:
            self.subscriptions[topic] = {}

        # Save the listener callback
        self.subscriptions[topic][subscriber.id] = subscriber.receive
        subscriber._send = self.inject

    def unsubscribe(self, subscriber_id, topic):

        """
        Removes a subscriber from a specific topic, effectively stopping event notifications for that topic.

        Parameters:
            subscriber_id (str): The unique ID of the subscriber to be removed.
            topic (str): The topic from which the subscriber is unsubscribing.
        """

        # If topic exists
        if topic in self.subscriptions:
            # If subscriber_id is a listener in the topic
            if subscriber_id in self.subscriptions[topic]:
                # Remove from the list
                del self.subscriptions[topic][subscriber_id]

    def inject(self, event):

        """
        Adds an event to the queue for processing. If the consume method is not currently processing events,
        it will start processing this event and any others in the queue.

        Parameters:
            event (Event): The event to be added to the queue. Must have a 'topic' and 'sender' attribute.
        """

        # Put event into the queue
        self.events.put(event)

        # If consume() is not running already
        if not self.lock:
            self.consume()

    # Single threaded because it is designed to be deterministic
    # It should only guarantee the order
    def consume(self):
        """
        Processes all events in the queue, ensuring each subscriber receives events for their subscribed topics.
        This method is designed to be deterministic, maintaining the order of events as they are processed.
        """

        self.lock = True

        while not self.events.empty():

            event = self.events.get()

            # List all callbacks functions subscribed in event.topic
            if event.topic in self.subscriptions:
                for subscriber in self.subscriptions[event.topic]:
                    # Do not send event to itself
                    if event.sender != subscriber:
                        # Call the function
                        self.subscriptions[event.topic][subscriber](event)

        self.lock = False
