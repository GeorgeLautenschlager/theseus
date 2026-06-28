class ChatObserver:
    def __init__(self, orient_chat_message_callback):
        self.orient_chat_message_callback = orient_chat_message_callback

    def observe_chat_message(self):
        """Block until the user sends a message on stdin, then invoke the callback."""
        message = input("User: ")
        self.orient_chat_message_callback(message)
