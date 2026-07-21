-- ioniq5-3.FEM을 불러온 뒤 그룹 1을 1도씩 회전시키며 해석하는 스크립트
-- FEMM Lua 콘솔에서 실행하면 됩니다.

open("ioniq5-3.FEM")

local group_id = 1
local start_angle = 0
local end_angle = 90
local step_deg = 10

-- 문제 정의 (FEM 파일에 이미 정보가 들어 있더라도 명시적으로 설정)
mi_probdef(0, "millimeters", "planar", 1e-8, 30)

print("angle_deg\ttorque_Nm")

for angle = start_angle, end_angle, step_deg do
    if angle > 0 then
        mi_selectgroup(group_id)
        mi_moverotate(0, 0, math.rad(step_deg))
        mi_clearselected()
    end

    mi_analyze()
    mi_loadsolution()

    -- 모든 선택 해제 후, 회전축 그룹의 블록만 선택
    mo_clearblock()
    mo_groupselectblock(group_id)

    -- 22: Steady-state torque
    local torque = mo_blockintegral(22)
    print(angle .. "\t" .. torque)
end

print("Analysis complete")