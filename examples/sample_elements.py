# 로케이터 컨텍스트 예시 (익명).
# collect_data.py가 이 클래스/속성 시그니처를 모델 프롬프트에 함께 주입해
# 모델이 존재하지 않는 로케이터를 생성하지 않도록 차단한다(grounding).
# 실제로는 android/ios 각각 elements.py에 화면 단위 클래스로 관리한다.

from appium.webdriver.common.appiumby import AppiumBy as By


class SideMenu:
    MENU_SETTINGS = (By.XPATH, '//android.widget.TextView[@text="설정"]')


class Settings:
    SETTINGS_NOTIFICATION = (By.XPATH, '//android.widget.TextView[@text="알림"]')


class NotificationSettings:
    sound_label  = (By.XPATH, '//android.widget.TextView[@text="효과음"]')
    sound_toggle = (By.XPATH, '//android.widget.TextView[@text="효과음"]/following-sibling::android.widget.Switch')
    sound_switch = (By.XPATH, '//android.widget.Switch[@content-desc="효과음"]')
