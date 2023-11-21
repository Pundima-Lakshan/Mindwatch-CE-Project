from vilt import ask_questions_video

# VIDEO_PATH = "D:\\Projects\\1 CEProject\\resources\\videos\\2.mp4"
QUESTION_TEXTS = "Is the person sleeping?"

def analyze_sleeping(VIDEO_PATH):
    ask_questions_video(VIDEO_PATH, QUESTION_TEXTS, "sleeping")

if __name__ == "__main__":
    VIDEO_PATH = "D:\\Projects\\1 CEProject\\resources\\videos\\2.mp4"
    analyze_sleeping(VIDEO_PATH)