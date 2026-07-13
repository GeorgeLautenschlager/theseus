class ChatEffector:
    def __init__(self):
        self.response = None

    def respond(self, response: str):
        """Send the response to the chat UI"""
        self.response = response
        print(f"Agent: {response}")

    def respond_callback(self, response: str):
        """Callback to be invoked by the cognitive core"""
        self.respond(response)
