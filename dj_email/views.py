import json

from django.shortcuts import render
from django.http import JsonResponse
from django.core import mail

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


def send_email(request):
    if request.method == 'POST':
        try:
            # 解析请求体
            body = json.loads(request.body.decode('utf-8'))

            # 验证请求数据
            is_valid, error_message = validate_email_data(body)
            if not is_valid:
                return JsonResponse({'code': 1, 'msg': error_message}, status=400)

            # 发送邮件
            mail.send_mail(
                subject=body.get('subject'),
                message=body.get('message'),
                from_email=settings.EMAIL_HOST_USER,
                recipient_list=body.get('recipient_list'),
            )

            # 记录日志
            # logger.info(f"Email sent to {body.get('recipient_list')} with subject: {body.get('subject')}")

            return JsonResponse({'code': 0, 'msg': '邮件发送成功'})
        except Exception as e:
            # 捕获异常并记录错误日志
            # logger.error(f"Failed to send email: {str(e)}")
            return JsonResponse({'code': 1, 'msg': f"邮件发送失败: {str(e)}"}, status=500)