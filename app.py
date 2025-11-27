# 1. IMPORTS
# We are bringing in the "Waiter" (Streamlit) and the tool to talk to the "Chef" (Google GenAI)
import streamlit as st
import google.generativeai as genai

# 2. PAGE SETUP
# This tells the browser what to display on the tab and the main title
st.set_page_config(page_title="Real Estate AI", page_icon="🏡")
st.title("🏡 AI Real Estate Listing Generator")
st.write("Enter the features of the house, and I will write a luxury description for you.")

# 3. API KEY CONFIGURATION
# We need to give our VIP Pass to the code so it can talk to Google.
# We will grab this secret key from Streamlit's secure storage later.
# specific "try/except" block prevents the app from crashing if the key is missing.
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except:
    st.error("API Key not found! Please set it in secrets.")

# 4. USER INPUTS
# We create a box where the user can type info.
# "features" is a variable that stores whatever the user types.
features = st.text_area("Property Features (e.g., 3 bedrooms, ocean view, renovated kitchen):")

# 5. THE BUTTON & THE LOGIC
# This 'if' statement only runs when the user clicks the button
if st.button("Generate Description"):
    
    # 6. THE PROMPT (The Order Ticket)
    # We combine your instructions with the user's input to create a specific request.
    prompt = f"""
    You are an expert real estate copywriter. 
    Write a captivating, luxurious instagram caption for a property with these features: {features}.
    Use emojis and hashtags.
    """

    # 7. CALLING THE CHEF
    # We load the specific AI model (gemini-1.5-flash is fast and cheap/free)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # We send the prompt to Google and wait for the response
    with st.spinner("Writing your listing..."):
        response = model.generate_content(prompt)

    # 8. SERVING THE DISH
    # We display the AI's answer on the screen
    st.success("Here is your listing:")

    st.write(response.text)
