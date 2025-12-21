#!/usr/bin/env python3
"""
Discord 알림 전송 스크립트
특수문자 이스케이프 문제를 해결하기 위해 Python으로 처리
"""

import json
import os
import sys
import requests
from datetime import datetime

def send_kr_notification(webhook_url: str, data_path: str) -> bool:
    """한국 주식 Discord 알림 전송"""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 시장 정보
        market = data.get('market_summary', {})
        stats = data.get('stats', {})
        recommendations = data.get('recommendations', {})

        kospi = market.get('kospi_index', 'N/A')
        kospi_chg = market.get('kospi_change_pct', 'N/A')
        kosdaq = market.get('kosdaq_index', 'N/A')
        kosdaq_chg = market.get('kosdaq_change_pct', 'N/A')

        strong_buy_count = stats.get('strong_buy', 0)
        buy_count = stats.get('buy', 0)

        # 상위 5개 적극매수 종목
        strong_buy_stocks = recommendations.get('strong_buy', [])[:5]
        top_stocks_text = "\n".join([
            f"• {s['name']} ({s['code']}) - 점수: {s['score']}"
            for s in strong_buy_stocks
        ]) or "추천 종목 없음"

        updated_at = data.get('updated_at', 'N/A')

        embed = {
            "title": "🇰🇷 한국 주식 오늘의 추천",
            "color": 3447003,  # Blue
            "fields": [
                {
                    "name": "📊 시장 현황",
                    "value": f"KOSPI: {kospi} ({kospi_chg}%)\nKOSDAQ: {kosdaq} ({kosdaq_chg}%)",
                    "inline": False
                },
                {
                    "name": "📈 추천 종목 수",
                    "value": f"적극매수: {strong_buy_count}개\n매수: {buy_count}개",
                    "inline": False
                },
                {
                    "name": "🚀 TOP 적극매수 종목",
                    "value": top_stocks_text,
                    "inline": False
                }
            ],
            "footer": {
                "text": f"마지막 업데이트: {updated_at}"
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        payload = {"embeds": [embed]}

        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code in [200, 204]:
            print(f"✅ 한국 주식 알림 전송 성공")
            return True
        else:
            print(f"❌ 한국 주식 알림 전송 실패: {response.status_code}")
            print(f"응답: {response.text}")
            return False

    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없음: {data_path}")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def send_us_notification(webhook_url: str, data_path: str) -> bool:
    """미국 주식 Discord 알림 전송"""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 시장 정보
        market = data.get('market_summary', {})
        stats = data.get('stats', {})
        recommendations = data.get('recommendations', {})

        sp500 = market.get('sp500_index', 'N/A')
        sp500_chg = market.get('sp500_change_pct', 'N/A')
        nasdaq = market.get('nasdaq_index', 'N/A')
        nasdaq_chg = market.get('nasdaq_change_pct', 'N/A')

        strong_buy_count = stats.get('strong_buy', 0)
        buy_count = stats.get('buy', 0)

        # 상위 5개 적극매수 종목
        strong_buy_stocks = recommendations.get('strong_buy', [])[:5]
        top_stocks_text = "\n".join([
            f"• {s['name']} ({s['code']}) - 점수: {s['score']}"
            for s in strong_buy_stocks
        ]) or "추천 종목 없음"

        updated_at = data.get('updated_at', 'N/A')

        embed = {
            "title": "🇺🇸 미국 주식 오늘의 추천",
            "color": 15844367,  # Gold
            "fields": [
                {
                    "name": "📊 시장 현황 (전일 종가)",
                    "value": f"S&P500: {sp500} ({sp500_chg}%)\nNASDAQ: {nasdaq} ({nasdaq_chg}%)",
                    "inline": False
                },
                {
                    "name": "📈 추천 종목 수",
                    "value": f"적극매수: {strong_buy_count}개\n매수: {buy_count}개",
                    "inline": False
                },
                {
                    "name": "🚀 TOP 적극매수 종목",
                    "value": top_stocks_text,
                    "inline": False
                }
            ],
            "footer": {
                "text": f"마지막 업데이트: {updated_at} | 미국장 개장 1시간 전"
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        payload = {"embeds": [embed]}

        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code in [200, 204]:
            print(f"✅ 미국 주식 알림 전송 성공")
            return True
        else:
            print(f"❌ 미국 주식 알림 전송 실패: {response.status_code}")
            print(f"응답: {response.text}")
            return False

    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없음: {data_path}")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def main():
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')

    if not webhook_url:
        print("❌ DISCORD_WEBHOOK_URL 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("사용법: python send_discord.py [kr|us|both]")
        sys.exit(1)

    region = sys.argv[1].lower()

    success = True

    if region in ['kr', 'both']:
        kr_path = 'data/kr/kr_recommendations.json'
        if os.path.exists(kr_path):
            if not send_kr_notification(webhook_url, kr_path):
                success = False
        else:
            print(f"⚠️ 한국 주식 데이터 없음: {kr_path}")

    if region in ['us', 'both']:
        us_path = 'data/us/us_recommendations.json'
        if os.path.exists(us_path):
            if not send_us_notification(webhook_url, us_path):
                success = False
        else:
            print(f"⚠️ 미국 주식 데이터 없음: {us_path}")

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
