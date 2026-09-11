-- 설정 변수 정의
local group_id = 10     -- 복사할 대상 그룹 번호 (원하는 그룹 번호로 수정)
local angle_step = 7.5 -- 회전 각도 (도)
local copy_count = 47  -- 복사 횟수

-- 1. 선택 초기화 및 그룹 선택
mi_clearselected()
mi_selectgroup(group_id)

-- 2. 7.5도 간격으로 47개 회전 복사 수행 (마지막 인자 4는 그룹 전체 선택 복사 비트마스크)
mi_copyrotate(0, 0, angle_step, copy_count, 4)

-- 3. 선택 해제
mi_clearselected()