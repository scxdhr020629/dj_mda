import json

from django.shortcuts import render
from django.http import JsonResponse
from django.core import mail
from django.core.mail import EmailMultiAlternatives  # 新增导入
from dj_mda import settings
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


# def send_email(request):
#     if request.method == 'POST':
#         try:
#             # 解析请求体
#             body = json.loads(request.body.decode('utf-8'))
#
#             # 验证请求数据
#             is_valid, error_message = validate_email_data(body)
#             if not is_valid:
#                 return JsonResponse({'code': 1, 'msg': error_message}, status=400)
#
#             # 发送邮件
#             mail.send_mail(
#                 subject=body.get('subject'),
#                 message=body.get('message'),
#                 from_email=settings.EMAIL_HOST_USER,
#                 recipient_list=body.get('recipient_list'),
#             )
#
#             # 记录日志
#             # logger.info(f"Email sent to {body.get('recipient_list')} with subject: {body.get('subject')}")
#             print("为什么前端没有json")
#             return JsonResponse({'code': 0, 'msg': '邮件发送成功', 'data': body})
#         except Exception as e:
#             # 捕获异常并记录错误日志
#             # logger.error(f"Failed to send email: {str(e)}")
#             return JsonResponse({'code': 1, 'msg': f"邮件发送失败: {str(e)}"}, status=500)

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