# DunClone

A Chrome extension that helps economics researchers analyze web content with a Cambridge economics professor's digital clone.

## Features

- Automatically detects economics-related content on webpages
- Displays a floating button on relevant pages
- Analyzes selected text or relevant page content
- Integrates with LLM (GPT-4o via OpenRouter) for insightful analysis
- Aesthetic inspired by Cambridge University colors
- Chat functionality to ask follow-up questions
- Animated UI elements for better user experience

## Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" using the toggle in the top right
3. Click "Load unpacked" and select the `chrome-extension` directory
4. The extension is now active and ready to use

## Usage

1. Click the extension icon and set your research objective in the popup
2. Browse the web - the extension will scan pages for economics content
3. When economics content is detected, a floating "D" button will appear
4. Click this button to open the sidebar analysis panel
5. You can also select specific text before clicking the button to analyze just that selection
6. The AI will analyze the content in relation to your research objective
7. Use the chat feature at the bottom of the sidebar to ask follow-up questions

## How It Works

1. **Content Detection**: The extension scans webpages for economics-related keywords
2. **Research Objective**: Your set objective is stored and used to contextualize AI analysis
3. **Content Analysis**: When triggered, the extension extracts relevant content and sends it with your objective to the LLM
4. **AI Analysis**: OpenRouter API (with GPT-4o) analyzes the content in relation to your research goals
5. **Results Display**: Analysis is presented in the sidebar for easy reference
6. **Chat Integration**: Continue the conversation with follow-up questions in the chat interface

## Code Structure

- `manifest.json`: Extension configuration and permissions
- `popup.html/js/css`: UI for setting research objectives
- `content.js/css`: Scripts for detecting content and injecting UI elements
- `background.js`: Handles API communication with OpenRouter and maintains chat history
- `sidebar.html/js/css`: UI for the analysis sidebar and chat functionality
- `.env`: Contains API key for OpenRouter

## Technical Details

- Uses Manifest V3 Chrome Extension API
- Integrates with OpenRouter to access GPT-4o
- Cambridge Blue color scheme
- Supports text selection for targeted analysis 