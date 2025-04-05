/**
 * Background script for DunClone
 * Handles communication between content scripts and the LLM API
 */

// API key for OpenRouter
const OPENROUTER_API_KEY = "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";

// Constants for API configuration
const USE_LOCAL_API = true; // Set to true to use local API, false for OpenRouter
const LOCAL_MODEL_PATH = '/mfs1/u/max/pollox-max/outputs/2025_04_05_23-08-12_0a9_model_for_rudolf/checkpoint_final';
const OPENROUTER_MODEL = 'openai/gpt-4o';

// API URLs
const LOCAL_API_URL = 'http://localhost:8000/v1';
const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';

// Headers
const OPENROUTER_HEADERS = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
  'HTTP-Referer': 'chrome-extension://dunclone',
  'X-Title': 'DunClone'
};

const LOCAL_API_HEADERS = {
  'Content-Type': 'application/json'
};

// Store conversation history
let conversationHistory = [];
let lastAnalysisContent = '';
let lastObjective = '';

/**
 * Formats the chat messages for OpenRouter API
 * @param {Array} messages - Array of message objects with role and content
 * @returns {Object} - Formatted request body
 */
function formatOpenRouterRequest(messages) {
  return {
    model: OPENROUTER_MODEL,
    messages: messages
  };
}

/**
 * Formats messages for local completions API
 * @param {Array} messages - Array of message objects with role and content
 * @returns {Object} - Formatted request body for completions API
 */
function formatLocalCompletionsRequest(messages) {
  // Convert chat messages to a single prompt string
  let promptText = "";
  
  messages.forEach(msg => {
    if (msg.role === 'system') {
      promptText += `System: ${msg.content}\n\n`;
    } else if (msg.role === 'user') {
      promptText += `User: ${msg.content}\n\n`;
    } else if (msg.role === 'assistant') {
      promptText += `Assistant: ${msg.content}\n\n`;
    }
  });
  
  // Add final prompt for assistant to continue
  promptText += "Assistant:";
  
  return {
    model: LOCAL_MODEL_PATH,
    prompt: promptText,
    max_tokens: 1000,
    temperature: 0.7
  };
}

/**
 * Parses the response from either API format
 * @param {Object} data - Response data from API
 * @param {boolean} isLocalAPI - Whether using local API
 * @returns {string} - Extracted response text
 */
function parseAPIResponse(data, isLocalAPI) {
  try {
    let responseText;
    
    if (isLocalAPI) {
      // Local completions API format
      if (!data.choices || data.choices.length === 0) {
        console.error('DunClone: Invalid response from local API:', data);
        return "Error: Invalid response from API. No choices returned.";
      }
      
      responseText = data.choices[0].text.trim();
    } else {
      // OpenRouter chat API format
      if (!data.choices || data.choices.length === 0 || !data.choices[0].message) {
        console.error('DunClone: Invalid response from OpenRouter API:', data);
        return "Error: Invalid response from API. No message returned.";
      }
      
      responseText = data.choices[0].message.content;
    }
    
    // Check if the model indicated it couldn't understand or process the content
    const errorIndicators = [
      "I apologize, but I don't see any content to analyze",
      "I don't have enough information",
      "I cannot access the content",
      "no content was provided",
      "unable to analyze the provided content"
    ];
    
    const hasError = errorIndicators.some(indicator => 
      responseText.toLowerCase().includes(indicator.toLowerCase())
    );
    
    if (hasError) {
      console.warn('DunClone: Model indicated it could not process the content:', responseText);
      responseText = `**Note: The model had difficulty analyzing the webpage content.**

${responseText}

**Additional Information:** 
If you're seeing this message, it means the model may not have received the full webpage content or had trouble processing it. You can try refreshing the page or selecting a different section of content.`;
    }
    
    return responseText;
  } catch (error) {
    console.error('DunClone: Error parsing API response:', error);
    return "Error parsing the response from the AI model.";
  }
}

/**
 * Sends a request to the appropriate API with the LLM prompt
 * 
 * @param {string} objective - The research objective
 * @param {string} content - The content to analyze
 * @returns {Promise<Object>} - The LLM response
 */
async function callLLM(objective, content) {
  try {
    // Store the objective for context in future chat
    lastObjective = objective;
    
    // Limit content length to prevent token overflow
    // Approximate tokens: ~3-4 chars per token, aim for ~3000 tokens max for content
    const MAX_CONTENT_LENGTH = 12000;
    let trimmedContent = content;
    
    if (content.length > MAX_CONTENT_LENGTH) {
      console.log(`DunClone: Trimming content from ${content.length} to ${MAX_CONTENT_LENGTH} characters`);
      
      // Instead of simple truncation, try to keep meaningful sections
      // First split by paragraph
      const paragraphs = content.split('\n\n');
      
      if (paragraphs.length > 1) {
        // Keep the beginning (title, metadata)
        let currentLength = 0;
        const importantParts = [];
        
        // Always include first few paragraphs which likely contain metadata
        for (let i = 0; i < Math.min(3, paragraphs.length); i++) {
          importantParts.push(paragraphs[i]);
          currentLength += paragraphs[i].length + 4; // +4 for '\n\n'
        }
        
        // Try to include any headings and their content, prioritizing h1, h2, h3
        const headingParagraphs = paragraphs.filter(p => 
          p.includes('H1:') || p.includes('H2:') || p.includes('H3:')
        ).slice(0, 10); // Limit to 10 major headings
        
        for (const p of headingParagraphs) {
          if (currentLength + p.length < MAX_CONTENT_LENGTH * 0.7) {
            importantParts.push(p);
            currentLength += p.length + 4;
          }
        }
        
        // Add more content until we reach the limit
        for (let i = 3; i < paragraphs.length; i++) {
          // Skip paragraphs we've already added from headings
          if (headingParagraphs.includes(paragraphs[i])) continue;
          
          // Check if adding this paragraph would exceed our limit
          if (currentLength + paragraphs[i].length > MAX_CONTENT_LENGTH) {
            // If we're getting close to the limit, add a note about truncation
            importantParts.push('... (content truncated due to length) ...');
            break;
          }
          
          importantParts.push(paragraphs[i]);
          currentLength += paragraphs[i].length + 4;
        }
        
        trimmedContent = importantParts.join('\n\n');
      } else {
        // Simple truncation if we can't split by paragraphs
        trimmedContent = content.substring(0, MAX_CONTENT_LENGTH) + '... (content truncated)';
      }
    }
    
    const prompt = `
    You are DunClone, a digital clone of a 19-year old economics prodigy helping analyze content from a webpage.

    User's Research Objective: ${objective}

    The user has navigated to a webpage with economics-related content. Analyze this content and provide insightful economic analysis. The content might be lengthy, so focus on the most relevant economics information.

    Content from Webpage:
    ${trimmedContent}

    Please provide your analysis using the following format:
    1. **Concise Summary of the Key Economic Points**
       - Present the most important economic ideas from the content
       - Use clear, academic language

    2. **Important Economic Concepts Identified**
       - List and briefly explain key economic concepts
       - Relate them to established economic theories where relevant

    3. **Critical Analysis of Economic Claims**
       - Evaluate the strength of economic arguments presented
       - Note any methodological concerns or limitations

    4. **Relevant Economic Context**
       - Provide additional context that enhances understanding
       - Connect to broader economic trends or research

    Use **bold** formatting for important terms and section headers. Format your response in clear, well-structured paragraphs. Avoid unnecessary explanations about the content's format.
    `;

    // Clear conversation history at the start of a new analysis
    conversationHistory = [
      { role: 'system', content: 'You are DunClone, a digital clone of a 19-year old economics prodigy helping analyze content from a webpage. You speak in a professional but warm academic tone and have deep expertise in economics.' },
      { role: 'user', content: prompt }
    ];

    let apiUrl, headers, requestBody;
    
    if (USE_LOCAL_API) {
      apiUrl = `${LOCAL_API_URL}/completions`;
      headers = LOCAL_API_HEADERS;
      requestBody = formatLocalCompletionsRequest(conversationHistory);
    } else {
      apiUrl = `${OPENROUTER_BASE_URL}/chat/completions`;
      headers = OPENROUTER_HEADERS;
      requestBody = formatOpenRouterRequest(conversationHistory);
    }

    console.log(`Making API request to: ${apiUrl}`);
    console.log('Request body size:', JSON.stringify(requestBody).length);

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('LLM API error:', errorData);
      throw new Error(`API error: ${response.status} - ${errorData.error?.message || 'Unknown error'}`);
    }

    const data = await response.json();
    const analysisContent = parseAPIResponse(data, USE_LOCAL_API);
    
    // Store the analysis for context in future chat
    lastAnalysisContent = analysisContent;
    
    // Add the assistant response to conversation history
    conversationHistory.push({ role: 'assistant', content: analysisContent });
    
    return {
      analysis: analysisContent
    };
  } catch (error) {
    console.error('Error calling LLM:', error);
    return {
      error: error.message || 'Failed to get analysis from LLM'
    };
  }
}

/**
 * Handles chat messages from the user
 * 
 * @param {string} message - The user's message
 * @returns {Promise<Object>} - The LLM response
 */
async function handleChatMessage(message) {
  try {
    // If this is the first chat message after a new analysis,
    // add a system message reminding of the context
    if (conversationHistory.length === 2 && lastAnalysisContent) {
      conversationHistory.push({ 
        role: 'system', 
        content: `Remember that you just analyzed some economics content with the objective: "${lastObjective}". Your last analysis was about: "${lastAnalysisContent.substring(0, 300)}...". The user's questions will be about this content and your analysis. If asked about something outside of the webpage, please use your knowledge to evaluate the question. Do not refuse to answer just because it is not in the text.` 
      });
    }
    
    // Add user message to conversation history
    conversationHistory.push({ role: 'user', content: message });
    
    let apiUrl, headers, requestBody;
    
    if (USE_LOCAL_API) {
      apiUrl = `${LOCAL_API_URL}/completions`;
      headers = LOCAL_API_HEADERS;
      requestBody = formatLocalCompletionsRequest(conversationHistory);
    } else {
      apiUrl = `${OPENROUTER_BASE_URL}/chat/completions`;
      headers = OPENROUTER_HEADERS;
      requestBody = formatOpenRouterRequest(conversationHistory);
    }

    console.log(`Making API request to: ${apiUrl}`);
    console.log('Request body:', requestBody);

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('LLM API error:', errorData);
      throw new Error(`API error: ${response.status} - ${errorData.error?.message || 'Unknown error'}`);
    }

    const data = await response.json();
    const responseContent = parseAPIResponse(data, USE_LOCAL_API);
    
    // Add the assistant response to conversation history
    conversationHistory.push({ role: 'assistant', content: responseContent });
    
    return {
      response: responseContent
    };
  } catch (error) {
    console.error('Error in chat:', error);
    return {
      error: error.message || 'Failed to process your message'
    };
  }
}

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('Background script received message:', message.type);
  
  if (message.type === 'ANALYZE_CONTENT') {
    // Extract data from the message
    const { objective, content } = message.data;
    
    // Call the LLM API and send the response back
    callLLM(objective, content)
      .then(result => {
        console.log('LLM response received');
        sendResponse(result);
      })
      .catch(error => {
        console.error('Error in LLM call:', error);
        sendResponse({ error: error.message || 'Failed to analyze content' });
      });
    
    // Return true to indicate we'll send a response asynchronously
    return true;
  } 
  else if (message.type === 'SEND_CHAT_MESSAGE') {
    // Handle chat message
    const { message: chatMessage } = message.data;
    
    handleChatMessage(chatMessage)
      .then(result => {
        console.log('Chat response received');
        sendResponse(result);
      })
      .catch(error => {
        console.error('Error in chat:', error);
        sendResponse({ error: error.message || 'Failed to process your message' });
      });
    
    // Return true to indicate we'll send a response asynchronously
    return true;
  }
}); 