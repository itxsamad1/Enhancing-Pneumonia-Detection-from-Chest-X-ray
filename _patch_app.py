import os

filepath = 'app.py'
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('''    if options.get("denoising", False):
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        applied_techniques.append("Denoising")''', 
'''    if options.get("denoising", False):
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        applied_techniques.append("Denoising")

    if options.get("sharpening", False):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        applied_techniques.append("Image Sharpening")''')

text = text.replace('''        if "denoising" not in st.session_state:
            st.session_state.denoising = False''', 
'''        if "denoising" not in st.session_state:
            st.session_state.denoising = False
        if "sharpening" not in st.session_state:
            st.session_state.sharpening = False''')

text = text.replace('''        denoising = st.checkbox("Apply Denoising",
                                value=st.session_state.denoising, key="denoise_cb")

        st.session_state.clahe = clahe
        st.session_state.histogram_eq = histogram_eq
        st.session_state.denoising = denoising''', 
'''        denoising = st.checkbox("Apply Denoising",
                                value=st.session_state.denoising, key="denoise_cb")
        sharpening = st.checkbox("Apply Image Sharpening",
                                 value=st.session_state.sharpening, key="sharpen_cb")

        st.session_state.clahe = clahe
        st.session_state.histogram_eq = histogram_eq
        st.session_state.denoising = denoising
        st.session_state.sharpening = sharpening''')

text = text.replace('''                st.session_state.denoising = True
                st.rerun()''', 
'''                st.session_state.denoising = True
                st.session_state.sharpening = True
                st.rerun()''')

text = text.replace('''                st.session_state.denoising = False
                st.rerun()''', 
'''                st.session_state.denoising = False
                st.session_state.sharpening = False
                st.rerun()''')

text = text.replace('''            "denoising": st.session_state.denoising,
        }''', 
'''            "denoising": st.session_state.denoising,
            "sharpening": st.session_state.sharpening,
        }''')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print("App patched successfully")
