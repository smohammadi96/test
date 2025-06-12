import gradio as gr
from chatbot_rag import Logger, LocalFarsiRAG  # فرض بر اینکه این کلاس‌ها در فایل chatbot_rag.py هستند

# ایجاد logger و RAG
logger = Logger()
rag = LocalFarsiRAG(logger=logger)

# بارگذاری فایل اسناد (فقط یک بار)
try:
    rag.load_document("merged.txt")  # مسیر به فایل متنی شما
except Exception as e:
    logger.log(f"Error loading document: {e}")

# تابع پاسخ‌دهی
def chat_with_bot(user_input):
    response = rag.ask(user_input)
    if "error" in response:
        return f"❌ خطا: {response['error']}"
    return response["answer"]

# رابط Gradio
demo = gr.Interface(
    fn=chat_with_bot,
    inputs=gr.Textbox(label="سوال خود را به فارسی وارد کنید"),
    outputs=gr.Textbox(label="پاسخ چت‌بات"),
    title="چت‌بات فارسی مبتنی بر اسناد",
    description="پاسخ به سوالات شما بر اساس اسناد بارگذاری‌شده. لطفاً فقط به فارسی سوال بپرسید.",
    theme="soft",
    allow_flagging="never"
)

# اجرا
if __name__ == "__main__":
    demo.launch()
