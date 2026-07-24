class TerminalChatObserver:
    def __init__(self, stimulus_log, orient_chat_message_callback):
        self.stimulus_log = stimulus_log
        self.orient_chat_message_callback = orient_chat_message_callback

    def observe_chat_message(self):
        """Block until the user sends a message on stdin, then invoke the callback."""
        message = input("User: ")
        self.stimulus_log.append(
            actor="user",
            type="chat_message",
            content={"message": message},
        )
        self.orient_chat_message_callback()
