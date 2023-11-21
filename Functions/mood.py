from Functions import vilt

ask_questions_video = vilt.ask_questions_video

# VIDEO_PATH = "D:\\Projects\\1 CEProject\\resources\\videos\\2.mp4"
QUESTION_TEXTS = "What is the mood of the person?"

def analyze_mood(VIDEO_PATH):
    ask_questions_video(VIDEO_PATH, QUESTION_TEXTS, "mood")

if __name__ == "__main__":
    VIDEO_PATH = "D:\\Projects\\1 CEProject\\resources\\videos\\2.mp4"
    analyze_mood(VIDEO_PATH)