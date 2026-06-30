# sample_tc.json(TC)을 입력하면 모델이 생성하는 Appium 코드 예시 (익명 합성).
# 학습 타깃은 def test_qa_{id}(driver, step_reporter): 함수 본문 하나.
# 주요 패턴: step_reporter.step()으로 스텝 구분, WebDriverWait로 대기,
# 공통 동작은 app_settings 헬퍼로 분리, 검증은 반드시 assert.
# (is_displayed만 호출하면 no-op이라 의미가 없어 assert를 붙여야 함)

def test_qa_10042(driver, step_reporter):

    ensure_home_screen(driver)
    wait = WebDriverWait(driver, 15)

    with step_reporter.step(0, "사전조건: 설정 > 알림 진입"):
        check_and_close_welcome_sheet(driver)
        open_side_menu(driver)
        wait.until(EC.element_to_be_clickable(SideMenu.MENU_SETTINGS)).click()
        time.sleep(0.3)
        wait.until(EC.element_to_be_clickable(Settings.SETTINGS_NOTIFICATION)).click()
        time.sleep(0.3)

    with step_reporter.step(1, "효과음 항목·토글 노출 확인"):
        assert_displayed(wait,
            (NotificationSettings.sound_label,  "효과음 항목"),
            (NotificationSettings.sound_toggle, "효과음 토글"),
        )

    with step_reporter.step(2, "토글 선택 시 ON/OFF 상태 반전 확인"):
        switch = wait.until(EC.presence_of_element_located(NotificationSettings.sound_switch))
        before = is_toggle_on(switch)
        wait.until(EC.element_to_be_clickable(NotificationSettings.sound_toggle)).click()
        time.sleep(0.4)
        after = is_toggle_on(wait.until(EC.presence_of_element_located(NotificationSettings.sound_switch)))
        assert after != before, f"토글 상태가 변하지 않음 (before={before}, after={after})"
