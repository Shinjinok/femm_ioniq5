-- 설정 값
group_num = 30     -- 복사할 원본 그룹 번호
dx = 16           -- X축 방향 이동 간격 (mm 또는 설정 단위)
dy = 0            -- Y축 방향 이동 간격
count = 7         -- 복사할 개수

for i = 1, count do
    -- 1. 원본 그룹 선택
    mi_selectgroup(group_num)
    
    -- 2. 지정된 간격만큼 평행이동 복사
    mi_movetranslateduplicate(dx, dy)
    
    -- 3. 새로 복사된 객체들의 그룹 번호를 0으로 해제 (중복 선택 방지)
    mi_setgroup(0)
    
    -- 4. 선택 해제
    mi_clearselected()
end