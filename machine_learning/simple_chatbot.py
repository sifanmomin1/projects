#!/usr/bin/env python3
"""
Simple Rule-Based Chatbot
- Pattern matching responses
- Conversation history
- Basic natural language processing
"""

import re
import random
import time
from datetime import datetime

class SimpleChatbot:
    """A simple rule-based chatbot implementation."""
    
    def __init__(self, name="ChatBot"):
        self.name = name
        self.conversation_history = []
        
        # Define response patterns
        self.patterns = [
            (r'hello|hi|hey', ['Hello!', 'Hi there!', 'Hey, how can I help you?']),
            (r'how are you', ['I\'m doing well, thanks for asking!', 'I\'m great! How are you?']),
            (r'what is your name', [f'My name is {name}.', f'I\'m {name}, nice to meet you!']),
            (r'bye|goodbye', ['Goodbye!', 'See you later!', 'Bye, have a great day!']),
            (r'thank you|thanks', ['You\'re welcome!', 'No problem!', 'Glad I could help!']),
            (r'what time is it', [self._get_time]),
            (r'what day is it', [self._get_date]),
            (r'help|command', ['I can respond to greetings, goodbyes, and questions about time, date, my name, and more.']),
            (r'weather', ['I\'m not connected to real-time weather data, but I hope it\'s nice outside!']),
            (r'tell me a joke', [
                'Why don\'t scientists trust atoms? Because they make up everything!',
                'Why did the scarecrow win an award? Because he was outstanding in his field!',
                'I told my wife she was drawing her eyebrows too high. She looked surprised.'
            ])
        ]
        
        # Default response when no pattern matches
        self.default_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "I don't have an answer for that yet.",
            "Interesting question! I'm still learning about that.",
            "I'm not programmed to respond to that. Can I help with something else?"
        ]
    
    def _get_time(self):
        """Return the current time."""
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."
    
    def _get_date(self):
        """Return the current date."""
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."
    
    def respond(self, user_input):
        """Generate a response based on the user input."""
        # Store the user input in conversation history
        self.conversation_history.append({"user": user_input, "timestamp": datetime.now()})
        
        # Clean user input
        user_input = user_input.lower().strip()
        
        # Find a matching pattern
        for pattern, responses in self.patterns:
            if re.search(pattern, user_input):
                # Select a random response
                response = random.choice(responses)
                
                # If response is a function, call it
                if callable(response):
                    response = response()
                
                # Store the response in conversation history
                self.conversation_history.append({"bot": response, "timestamp": datetime.now()})
                return response
        
        # If no pattern matches, return a default response
        default_response = random.choice(self.default_responses)
        self.conversation_history.append({"bot": default_response, "timestamp": datetime.now()})
        return default_response
    
    def get_conversation_history(self):
        """Return the conversation history."""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        return "Conversation history cleared."

def chat_session():
    """Start an interactive chat session with the chatbot."""
    chatbot = SimpleChatbot("Buddy")
    print(f"{chatbot.name}: Hello! I'm {chatbot.name}. Type 'bye' to exit.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print(f"{chatbot.name}: {chatbot.respond(user_input)}")
            break
        
        response = chatbot.respond(user_input)
        print(f"{chatbot.name}: {response}")

if __name__ == "__main__":
    chat_session()
