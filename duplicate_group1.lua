-- 그룹 1인 대상을 원점을 중심으로 90도 회전하여 1개 복재하는 스크립트
-- FEMM Lua Console에서 실행: dofile("duplicate_group1.lua")

open("ioniq5-4.FEM")

local group_id = 1
local angle_rad =45 -- 90도(π/2)

mi_probdef(0, "millimeters", "planar", 1e-8, 30)

mi_selectgroup(group_id)
mi_copyrotate(0, 0, angle_rad, 1)
mi_clearselected()

mi_zoomnatural()
print("Group 1 copied once at a 90-degree rotation around the origin.")
