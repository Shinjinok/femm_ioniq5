-- 변경할 그룹 번호 지정 (예: 8번 그룹)
local target_group = 1
local new_material = "N45"

mi_clearselected()
-- 해당 그룹에 속한 모든 요소 선택
mi_selectgroup(target_group)

-- 선택된 모든 블록의 재질을 N45로 일괄 변경
-- (automesh=0, meshsize=0, circuit="", magdir=0, group=target_group, turns=0 유지)
mi_setblockprop(new_material, 0, 0, "", 0, target_group, 0)

mi_clearselected()
print("지정된 그룹의 자석 재질이 " .. new_material .. "(으)로 일괄 변경되었습니다.")