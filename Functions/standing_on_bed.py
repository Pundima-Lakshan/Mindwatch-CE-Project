from Functions import vilt

ask_questions_video = vilt.ask_questions_video

# VIDEO_PATH = "D:\\Projects\\1 CEProject\\resources\\videos\\2.mp4"
QUESTION_TEXTS = "Is the person standing on a bed?"

def analyze_standing_on_bed(VIDEO_PATH):
    ask_questions_video(VIDEO_PATH, QUESTION_TEXTS, "standing_on_bed")

if __name__ == "__main__":
    VIDEO_PATH = "D:\\Projects\\1 CEProject\\resources\\videos\\2.mp4"
    analyze_standing_on_bed(VIDEO_PATH)