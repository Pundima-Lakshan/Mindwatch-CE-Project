from vilt import ask_question_image, ask_questions_video

VIDEO_PATH = "D:\\Projects\\1 CEProject\\resources\\videos\\2.mp4"
QUESTION_TEXTS = "Is the person sleeping?"

ask_questions_video(VIDEO_PATH, QUESTION_TEXTS)