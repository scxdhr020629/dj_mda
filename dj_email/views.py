import json

from django.shortcuts import render
from django.http import JsonResponse
from django.core import mail
from django.core.mail import EmailMultiAlternatives  # 新增导入
from openai.types import Reasoning
from django.views.decorators.csrf import csrf_exempt
from dj_mda import settings
from openai import OpenAI
# import logging

# Create your views here.
def hello(request):
    return JsonResponse({'hello': 'world'})


# 配置日志
# logger = logging.getLogger(__name__)


def validate_email_data(body):
    """验证邮件数据是否合法"""
    required_fields = ['subject', 'message', 'recipient_list']
    for field in required_fields:
        if field not in body or not body[field]:
            return False, f"{field} is missing or empty"
    return True, None


@csrf_exempt
def test_openai(request):
    """
    API endpoint to interact with OpenAI model via POST request.

    Returns both content and reasoning_content in the response data.

    POST Parameters:
        message (str): The message/query to send to the AI model
    """
    # Ensure the request is a POST
    if request.method != 'POST':
        return JsonResponse({
            'code': 1,
            'msg': 'Only POST requests are allowed',
            'data': None
        }, status=405)

    try:
        # Parse the JSON data from the request body
        data = json.loads(request.body)

        # Extract the message from the request data
        user_message = data.get('message')

        # Validate input
        if not user_message or not isinstance(user_message, str):
            return JsonResponse({
                'code': 1,
                'msg': 'A valid "message" field is required',
                'data': None
            }, status=400)

        # Initialize the OpenAI client
        client = OpenAI(
            api_key="bce-v3/ALTAK-sG68G5A0VyOCbSaOTlG1n/4507349759e20b348f7829576282c8be48e5a226",
            base_url="https://qianfan.baidubce.com/v2",
        )

        # Prepare messages for the model
        messages = [{"role": "user", "content": user_message}]

        # Call the OpenAI API
        response = client.chat.completions.create(
            model="deepseek-r1-distill-qwen-32b",
            messages=messages
        )

        # Extract content and reasoning_content
        content = response.choices[0].message.content
        reasoning_content = response.choices[0].message.reasoning_content if hasattr(response.choices[0].message,
                                                                                     'reasoning_content') else None

        # Print for debugging purposes
        print("hello")
        print(f"Content: {content}")
        print(f"Reasoning Content: {reasoning_content}")

        # Return both content and reasoning_content in data field
        return JsonResponse({
            'code': 0,
            'msg': '发送成功',
            'data': {
                'reasoning_content': reasoning_content,
                'content': content
            }
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'code': 1,
            'msg': 'Invalid JSON data in request body',
            'data': None
        }, status=400)
    except Exception as e:
        # Log the exception for debugging
        print(f"Error in test_openai: {str(e)}")

        return JsonResponse({
            'code': 1,
            'msg': '处理请求时发生错误',
            'data': None
        }, status=500)




def send_email(request):
    if request.method == 'POST':
        try:
            # 解析请求体
            body = json.loads(request.body.decode('utf-8'))

            # 验证请求数据
            is_valid, error_message = validate_email_data(body)
            if not is_valid:
                return JsonResponse({'code': 1, 'msg': error_message}, status=400)

            subject = body.get('subject')
            message = body.get('message')
            recipient_list = body.get('recipient_list')

            # 检查是否为HTML邮件
            is_html = body.get('is_html', False) or body.get('content_type') == 'text/html'

            if is_html:
                # 如果是HTML格式，使用EmailMultiAlternatives
                # 创建邮件对象 (必须提供一个纯文本版本作为备用)
                plain_text = "请使用支持HTML的邮件客户端查看此邮件。"
                email = EmailMultiAlternatives(
                    subject=subject,
                    body=plain_text,  # 纯文本版本
                    from_email=settings.EMAIL_HOST_USER,
                    to=recipient_list
                )

                # 添加HTML内容
                email.attach_alternative(message, "text/html")

                # 发送邮件
                email.send()
            else:

                # 如果是普通文本邮件，使用原来的方法
                mail.send_mail(
                    subject=subject,
                    message=message,
                    from_email=settings.EMAIL_HOST_USER,
                    recipient_list=recipient_list,
                )

            # 记录日志
            print(f"Email sent to {recipient_list} with subject: {subject}")
            return JsonResponse({'code': 0, 'msg': '邮件发送成功', 'data': body})
        except Exception as e:
            # 捕获异常并记录错误日志
            print(f"Failed to send email: {str(e)}")
            return JsonResponse({'code': 1, 'msg': f"邮件发送失败: {str(e)}"}, status=500)