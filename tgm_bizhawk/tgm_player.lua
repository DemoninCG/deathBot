-- tgm_player.lua -- state publisher + input playback for the TGM2 bot.
--
-- Protocol (true IRS via ARE hold, probe-verified: taps do nothing, a hold
-- rotates exactly once):
--   * lock publishes (quiescence) refresh the board; Python picks the IRS
--     ROOT for the upcoming piece and writes a hold move;
--   * Lua HOLDS that button every frameend through spawn+1, then releases;
--   * spawn publishes carry the held button (header.held, authoritative);
--   * Python decides the full placement constrained to that root with the
--     true next, Lua plays the steps from adoption+1 with Down to lock.
-- Diagnostics: ORI spawncheck snapshots every spawned orientation (audits
-- the hold offline); lock publishes never decide, spawn publishes never
-- hold. State files carry frame/cnt/cur/next/tmr/why/exec/held.
--
-- Playback: one input per STEP_FRAMES (3f holds reliably, 1f presses
-- demonstrably miss polls; 2f validated live through Death 500+). Long
-- paths may lock early at high levels (lock delay drops 30->26->22) —
-- the abort keeps leftovers from corrupting the next piece, and the
-- mirror shows the divergence.
--
-- CONFIG.MAP translates sim input names to YOUR BizHawk buttons.
-- Probe-verified: Button 1 = CCW (TGM A), Button 2 = CW (TGM B),
-- Button 3 = CCW duplicate, Left/Right = 1 cell per tap, Up = sonic
-- (no-op when grounded). Down locks a grounded piece on the spot
-- (frame-by-frame verified); playback ends every plan with one Down press.
-- Read-only except joypad.set while a plan is executing; never System Bus.

local CONFIG = {
  -- 2f step spacing: tap/release alternation never charges DAS (no
  -- auto-shift), and 1f presses demonstrably miss input polls. WARNING:
  -- consecutive same-direction presses read as a HOLD under DAS; watch
  -- for runaway slides at 500+ (DAS shortens further there).
  STEP_FRAMES = 2,
  MOVES = "tgm_moves.txt",
  MAP = { Left = "Left", Right = "Right", Down = "Down",
          CW = "Button 2", CCW = "Button 1" },
};

local RAM = "sh2 : ram : 0x6000000-0x60FFFFF";
local SPR = "sh2 : ram : 0x4000000-0x400FFFF";
local OUTFILE = "tgm_state.txt";

-- Perf: orientation trace (ORI spawncheck + per-rotation snapshots) is a
-- diagnostic that costs 2+ full sprite-list scans per piece plus console
-- spam. It does not affect decisions or playback. Keep OFF for real-time
-- play/recording; set true only when auditing holds offline.
local ORI_ENABLED = false;

-- Perf: localize the hot memory readers once. The per-read pcall()s below
-- used to cost ~4000 protected calls per board scan (up to ~10 scans per
-- piece); a single pcall around the whole scan preserves the nil,err
-- contract at ~1/4000th of the cost. The outer onframeend pcall remains
-- as the final backstop.

local LOCK_COLR = { [0x3F] = "I", [0x5D] = "J", [0x67] = "L", [0x71] = "O",
                    [0x53] = "S", [0x7B] = "T", [0x49] = "Z" };
-- Move steps arrive from Python in TGM-side direction names (the bridge
-- translates per piece: sim "RotateCw" on T/J/L means TGM-CCW and vice
-- versa, verified cell-by-cell). Valid step names: Left/Right/CW/CCW.
local STEP_BTN = { Left = "Left", Right = "Right", CW = "CW", CCW = "CCW" };

local function sext(v, bits)
  if v >= (2 ^ (bits - 1)) then return v - (2 ^ bits) else return v end
end

local _read_u16 = memory.read_u16_be;
local _read_u32 = memory.read_u32_be;

local function scan_board()
  -- Single-pcall wrapper: the inner scan throws on a bus error, we
  -- convert to the historical (nil, err) return. Normal path: zero
  -- pcalls inside the 1024-entry loop (was 4 per entry).
  local function inner()
    local grid = {};
    for r = 0, 19 do grid[r] = {} end
    for i = 0, 0x3FF do
      local e = _read_u16(0x3800 + i * 2, SPR);
      local base = ((e & 0x3FF) * 4) * 4;
      -- Read w1 first: most non-field sprites are not 1x1, so skipping
      -- w0/w2 there saves ~2 bus reads per such entry. Field cells need
      -- all three words.
      local w1 = _read_u32(base + 4, SPR);
      local wide = (math.floor(w1 / 256) & 0xF) + 1;
      local high = (math.floor(w1 / 16777216) & 0xF) + 1;
      if wide == 1 and high == 1 then
        local w0 = _read_u32(base, SPR);
        local w2 = _read_u32(base + 8, SPR);
        local tnum = w2 & 0x7FFFF;
        if tnum >= 0x18200 and tnum <= 0x182FF then
          local colr = math.floor(w2 / 16777216) & 0xFF;
          if LOCK_COLR[colr] ~= nil then
            local x = sext(w0 & 0x3FF, 10);
            local y = sext(math.floor(w0 / 65536) & 0x3FF, 10);
            if (x - 18) % 8 == 0 and (y - 41) % 8 == 0 then
              local c = math.floor((x - 18) / 8);
              local r = math.floor((y - 41) / 8);
              if c >= 0 and c < 10 and r >= 0 and r < 20
                 and grid[r][c] == nil then
                grid[r][c] = LOCK_COLR[colr];
              end
            end
          end
        end
      end
      if (e & 0x4000) ~= 0 then break end
    end
    local rows = {};
    for r = 0, 19 do
      local row = {};
      for c = 0, 9 do row[#row + 1] = grid[r][c] or "." end
      rows[#rows + 1] = table.concat(row);
    end
    return rows;
  end
  local ok, res = pcall(inner);
  if ok then return res end
  return nil, "sprite read failed: " .. tostring(res);
end

-- On-grid mino cells "c_r=COLR" for ANY 1x1 0x182xx sprite. Falling pieces
-- use piece-specific animated palettes (T: 0x7E/0x7D, I: 0x3E/0x42, …),
-- so callers subtract the cached locked board to isolate the live piece.
-- DIAGNOSTIC ONLY (gated by ORI_ENABLED): same single-pcall treatment as
-- scan_board; untouched semantics when enabled.
local function snapshot_minos()
  local function inner()
    local cells = {};
    for i = 0, 0x3FF do
      local e = _read_u16(0x3800 + i * 2, SPR);
      local base = ((e & 0x3FF) * 4) * 4;
      local w1 = _read_u32(base + 4, SPR);
      local wide = (math.floor(w1 / 256) & 0xF) + 1;
      local high = (math.floor(w1 / 16777216) & 0xF) + 1;
      if wide == 1 and high == 1 then
        local w0 = _read_u32(base, SPR);
        local w2 = _read_u32(base + 8, SPR);
        local tnum = w2 & 0x7FFFF;
        if tnum >= 0x18200 and tnum <= 0x182FF then
          local colr = math.floor(w2 / 16777216) & 0xFF;
          local x = sext(w0 & 0x3FF, 10);
          local y = sext(math.floor(w0 / 65536) & 0x3FF, 10);
          if (x - 18) % 8 == 0 and (y - 41) % 8 == 0 then
            local c = math.floor((x - 18) / 8);
            local r = math.floor((y - 41) / 8);
            if c >= 0 and c < 10 and r >= 0 and r < 20 then
              cells[c .. "_" .. r] = colr;
            end
          end
        end
      end
      if (e & 0x4000) ~= 0 then break end
    end
    return cells;
  end
  local ok, res = pcall(inner);
  if ok then return res end
  return nil;
end

-- publish state (dedupe by board+cnt/cur/nxt); returns publish frame or nil.
-- rows come from the caller so lock scans (fresh) and spawn reuses (cached)
-- stay distinct. Cache updates are the caller's job (lock scans only).
local lastkey = nil;
local lastpub = nil;
-- lastmis/lastnomoves: dedupe keys for the poll diagnostics below (one
-- line per publish cycle / 600f — never per-frame spam).
local lastmis = nil;
local lastnomoves = -100000;
local exec = "idle";

local function board_key(rows, cnt, cur, nxt)
  return string.format("%d/%d/%d/", cnt, cur, nxt)
    .. table.concat(rows, "/");
end

-- A board is "settled" when it holds no transients: no FULL row (line
-- clear in progress — rows vanish during the delay) and no fully-EMPTY
-- row with filled cells above it (clear in progress, mid-drop: cleared
-- lines disappear first, rows above fall after a delay). Real boards can't
-- hold either: full rows always clear, and a completely empty row under
-- filled cells cannot persist (partial holes can and do — those are fine).
-- Unsettled boards are never published for decisions and never cached;
-- quiescence simply waits them out (cap failsafe below).
local function board_settled(rows)
  local filled_above = false;
  for r = 0, 19 do
    local row = rows[r + 1];
    local filled, empty = 0, 0;
    for c = 1, 10 do
      if row:sub(c, c) == "." then empty = empty + 1
      else filled = filled + 1 end
    end
    if filled == 10 then return false end;
    if empty == 10 and filled_above then return false end;
    if filled > 0 then filled_above = true end
  end
  return true;
end

local function publish_rows(rows, cnt, cur, nxt, tmr, why, held)
  local key = board_key(rows, cnt, cur, nxt);
  if key == lastkey then return nil end
  lastkey = key;
  local frame = emu.framecount();
  -- exec values "play i/n" and "hold Button N" contain spaces, which the
  -- Python header parser (whitespace-split) rejects as "bad header" — and
  -- every such publish then skips its decision (lock publishes mid-path
  -- never choose an IRS root: the "holds never engage" starvation).
  -- Sanitize here; the mirror also parses tolerantly (either form reads).
  local head = string.format(
    "frame=%d cnt=%d cur=%d next=%d tmr=%d why=%s exec=%s held=%s\n",
    frame, cnt, cur, nxt, tmr, why, (exec:gsub("%s+", "_")),
    held or "-");
  local tmp = OUTFILE .. ".tmp";
  local fh = io.open(tmp, "w");
  if not fh then
    console.log("PLAYER ERROR: cannot write " .. tmp); return nil;
  end
  fh:write(head);
  for r = 0, 19 do fh:write(rows[r + 1] .. "\n") end
  fh:close();
  local okren = false;
  if os ~= nil and os.rename ~= nil then
    -- Windows rename fails when the destination exists (i.e. every publish
    -- after the first), which used to fall through to a torn direct write
    -- that Python could catch mid-flight ("bad shape"/truncated header).
    -- Remove-then-rename keeps the swap atomic; the brief missing-file
    -- window just reads as "file not found" and retries next poll.
    if os.remove ~= nil then pcall(os.remove, OUTFILE) end
    local ok, res = pcall(os.rename, tmp, OUTFILE);
    okren = ok and res;
  end
  if not okren then
    local fh2 = io.open(OUTFILE, "w");
    if not fh2 then
      console.log("PLAYER ERROR: cannot write " .. OUTFILE); return nil;
    end
    fh2:write(head);
    for r = 0, 19 do fh2:write(rows[r + 1] .. "\n") end
    fh2:close();
  end
  lastpub = frame;
  console.log(string.format("PLAYER: wrote %s (cnt=%d cur=%d next=%d %s)",
    OUTFILE, cnt, cur, nxt, why));
  return frame;
end

-- adopted plan for the currently falling piece (nil when none).
local plan = nil;      -- {steps, idx, next_frame, born_frame}
-- consumed_for: for_frame of the already-adopted play plan (cleared on
-- turnover). Each plan is adopted ONCE: re-adopting the same moves file
-- replays the path + Down every few frames, and every Down re-arms
-- quiescence — short/0-step loops re-arm faster than a stable pair can
-- form, so lock publishes (and the board cache) starve forever.
local consumed_for = nil;
-- ARE hold state (true IRS): hold_btn = mapped button or nil; release at
-- rel_at (turnover+1); hold_since/cap backstop; hold_dir reports the
-- TGM-side direction in state headers for Python's root adjudication.
local hold_btn = nil;
local hold_dir = nil;  -- TGM-side direction actually held ("CW"/"CCW")
local hold_since = 0;
local rel_at = nil;
-- spawn orientation check (diagnostic): snapshot 2f after turnover.
local ori_at = nil;
-- done_hold: path complete + Down pressed, piece locks on command.
-- Keep buttons released (and user input blocked) until the turnover.
local done_hold = false;
-- set after a press the scheduler won't itself release (final Down): the
-- next frameend releases first. NEVER press-then-release in one frameend:
-- the last joypad.set wins, which silently canceled the old final Down.
local need_release = false;
-- lock quiescence: the locking piece converts to locked colors a few frames
-- after the lock edge (an immediate scan sees the pre-lock board and the
-- dedupe would suppress the lock publish forever). Re-scan every few frames
-- while timer==30; each new board state publishes. Spawn reuses the cached
-- lock board (never rescans: the falling piece must not pollute the model).
local board_cache = nil;   -- rows from the latest lock scan that published
local lock_watch = false;
local lock_frame = 0;
local next_qscan = 0;
local last_qkey = nil;
-- orientation trace: when true, the next frameend snapshots the live piece
-- (on-grid minos minus cached locked cells) and logs it. Armed at adoption
-- (spawn orientation) and after every rotation step, so each rotation can
-- be checked against the sim offline. ~2 extra scans per piece max.
local verify_armed = nil;  -- label string or nil

-- Falling-piece cells as "c_r=COLR ..." (sorted), or nil on read failure.
local function active_cells()
  local all = snapshot_minos();
  if all == nil then return nil end
  local out = {};
  for k, co in pairs(all) do
    local c, r = k:match("(%d+)_(%d+)");
    c, r = tonumber(c), tonumber(r);
    local locked = false;
    if board_cache ~= nil and board_cache[r + 1] ~= nil then
      locked = board_cache[r + 1]:sub(c + 1, c + 1) ~= ".";
    end
    if not locked then out[#out + 1] = string.format("%s=%02X", k, co) end
  end
  table.sort(out);
  return table.concat(out, " ");
end

local function read_moves()
  -- 4-line format (atomic via tmp+rename on the Python side):
  --   phase=hold|play
  --   for_frame=N  (must equal lastpub: the state Python decided on)
  --   hold=CW|CCW|-  (ARE hold button, TGM-side; "-" on play moves)
  --   steps=a,b,c  (empty on hold moves)
  local fh = io.open(CONFIG.MOVES, "r");
  if not fh then return nil end
  local l1, l2, l3, l4 = fh:read("*l"), fh:read("*l"), fh:read("*l"), fh:read("*l");
  fh:close();
  if l1 == nil or l2 == nil then return nil end
  local phase = l1:match("^phase=(.+)$");
  if phase ~= "hold" and phase ~= "play" then return nil end
  local fr = l2:match("^for_frame=(%d+)$");
  -- NOTE: Lua patterns have no "|" alternation; match openly, validate below.
  local hold = l3 and l3:match("^hold=(.+)$") or nil;
  if hold ~= "-" and hold ~= "CW" and hold ~= "CCW" then hold = nil end
  local steps = {};
  if l4 ~= nil then
    local s = l4:match("^steps=(.*)$");
    if s == nil then return nil end
    if s ~= "" then
      for name in s:gmatch("[^,]+") do
        if STEP_BTN[name] == nil then return nil end
        steps[#steps + 1] = name;
      end
    end
  end
  if fr == nil or hold == nil then return nil end
  return { phase = phase, for_frame = tonumber(fr),
           hold = hold, steps = steps };
end

local function release_all()
  -- revert every known button to user control (nil = no override).
  joypad.set({ ["Button 1"] = nil, ["Button 2"] = nil, ["Button 3"] = nil,
               ["Down"] = nil, ["Left"] = nil, ["Right"] = nil,
               ["Up"] = nil }, 1);
end

local function press_mapped(name)
  -- name: TGM-side direction ("Left"/"Right"/"CW"/"CCW"/"Down").
  local btn = CONFIG.MAP[name];
  if btn == nil then return end
  joypad.set({ [btn] = true }, 1);
end

local function hold_release(skip)
  -- force-release mapped buttons (also blocks physical interference),
  -- except an actively held IRS button.
  local t = {};
  for _, b in pairs(CONFIG.MAP) do
    if b ~= skip then t[b] = false end
  end
  joypad.set(t, 1);
end

local lastcnt, lastcur, lastnxt, prevtmr = nil, nil, nil, nil;
local frames = 0;

event.onframeend(function()
  local ok, err = pcall(function()
    frames = frames + 1;
    local frame = emu.framecount();
    local okc, cnt = pcall(memory.read_u8, 0x64C16, RAM);
    local okt, tmr = pcall(memory.read_u8, 0x64C35, RAM);
    local oku, cur = pcall(memory.read_u8, 0x64C05, RAM);
    local okn, nxt = pcall(memory.read_u8, 0x64C04, RAM);
    local okl, lockval = pcall(memory.read_u8, 0x64BE0, RAM);
    if not (okc and okt and oku and okn and okl) then
      console.log("PLAYER ERROR: main-RAM read failed"); return;
    end
    if frames % 600 == 0 then
      console.log(string.format(
        "PLAYER: alive f=%d cnt=%d tmr=%d cur=%d next=%d exec=%s",
        frame, cnt, tmr, cur, nxt, exec));
    end
    if lastcnt == nil then
      lastcnt, lastcur, lastnxt, prevtmr = cnt, cur, nxt, tmr; return;
    end
    if need_release then release_all(); need_release = false end
    local turnover = (cur ~= lastcur) or (nxt ~= lastnxt);

    -- orientation trace (see declarations): snapshot before any pressing
    -- this frameend, so the reading reflects the previous frame's result.
    -- Covers both explicit checks (spawn/rotN) and the automatic spawncheck
    -- 2f after every turnover (audits what the ARE hold produced).
    -- Perf: gated by ORI_ENABLED (diagnostic only: 1-2 full sprite scans
    -- plus console spam per piece when on).
    if ORI_ENABLED and (verify_armed ~= nil or (ori_at ~= nil and frame >= ori_at)) then
      local ac = active_cells();
      console.log(string.format("ORI %s cur=%d [%s]",
        verify_armed or "spawncheck", cur, ac or "READ-FAILED"));
      verify_armed = nil; ori_at = nil;
    elseif not ORI_ENABLED then
      verify_armed = nil; ori_at = nil;
    end

    -- ARE hold (true IRS): press every frameend while set. Covers the
    -- spawn frameend too (release lands at rel_at, turnover+1).
    if hold_btn ~= nil then
      if rel_at ~= nil and frame >= rel_at then
        release_all(); hold_btn = nil; hold_dir = nil; rel_at = nil;
        if exec:sub(1, 4) == "hold" then exec = "idle" end
      elseif hold_since ~= nil and frame - hold_since > 120 then
        release_all(); hold_btn = nil; hold_dir = nil; rel_at = nil;  -- safety cap
        console.log("PLAYER: hold safety release");
      else
        joypad.set({ [hold_btn] = true }, 1);
      end
    end

    -- 1. turnover: abort a stale plan (early lock). Adoption happens in
    -- section 4 below, never here, so a plan always belongs to the piece
    -- that is currently falling (diag: adopting here played every plan
    -- one piece late).
    if turnover then
      if plan ~= nil and frame ~= plan.born_frame then
        console.log("PLAYER: abort plan (early lock?)");
        plan = nil; release_all();
      end
      -- keep holding through the spawn frameend (IRS samples held input
      -- at spawn); release lands at rel_at below. The spawncheck audits
      -- the produced orientation 2f later (ORI_ENABLED only).
      if hold_btn ~= nil and rel_at == nil then rel_at = frame + 1 end
      if ORI_ENABLED then ori_at = frame + 2 end
      done_hold = false;
      consumed_for = nil;  -- new piece: its play move is adoptable again
      if plan == nil and hold_btn == nil then exec = "idle" end
    end

    -- 2/3. step scheduler. (IRS holds are handled above, never here.)
    if plan ~= nil then
      -- 3. step scheduler.
      if frame >= plan.next_frame then
        if plan.idx <= #plan.steps then
          press_mapped(plan.steps[plan.idx]);
          exec = string.format("play %d/%d", plan.idx, #plan.steps);
          if ORI_ENABLED and (plan.steps[plan.idx] == "CW"
             or plan.steps[plan.idx] == "CCW") then
            verify_armed = "rot" .. plan.idx;
          end
          plan.idx = plan.idx + 1;
          plan.next_frame = frame + CONFIG.STEP_FRAMES;
        else
          -- path complete: Down locks a grounded piece on the spot
          -- (frame-by-frame verified). Release happens next frameend:
          -- releasing in this same frameend would cancel the press.
          -- Arm quiescence here too: a Down-lock never raises the timer
          -- edge, so without this the board cache would go stale forever.
          joypad.set({ [CONFIG.MAP.Down] = true }, 1);
          need_release = true;
          lock_watch = true; lock_frame = frame; next_qscan = frame + 2;
          last_qkey = nil;
          console.log("PLAYER: lock (Down)");
          plan = nil; done_hold = true; exec = "done";
        end
      else
        hold_release(hold_btn);
      end
    elseif plan == nil then
      -- adopt/hold Python's decision when it matches the latest publish.
      -- NOTE: no done_hold shadow here — holds must start while done_hold
      -- is still true from the path that just completed (diag: the shadow
      -- starved every hold; ROOT lines flowed but Lua never held).
      -- done_hold only gates the release-all below, exempting the hold.
      local m = nil;
      if not turnover then
        m = read_moves();
      end
      if m ~= nil and m.for_frame == lastpub then
        -- Hold moves are written exclusively on lock publishes, and the
        -- for_frame match pins adoption to that piece's post-lock window
        -- (no falling piece exists between a lock publish and the next
        -- turnover) — so there is no timer gate on holds. The old
        -- `tmr >= lockval` gate never opened after a Down-lock (timer
        -- frozen small, ARE too short to reach lockval: 0/8 holds
        -- engaged, tmr read 12-27 every poll) and starved every hold.
        if m.phase == "hold" then
          if m.hold ~= "-" then
            local btn = CONFIG.MAP[m.hold];
            if btn ~= nil then
              if hold_btn ~= btn then
                release_all();
                console.log("PLAYER: holding " .. btn .. " for IRS");
              end
              hold_btn, hold_dir, hold_since = btn, m.hold, frame;
              rel_at = nil;
              done_hold = false;  -- new cycle owns the buttons now
              exec = "hold " .. btn;
              -- press immediately: the next emulated frame may already be
              -- the spawn frame, and only frames covered by a press count.
              joypad.set({ [btn] = true }, 1);
            end
          elseif hold_btn ~= nil then
            release_all(); hold_btn = nil; hold_dir = nil; rel_at = nil;
          end
        elseif m.phase == "play" and tmr < lockval
           and m.for_frame ~= consumed_for then
          if hold_btn ~= nil then
            -- stale hold from a previous cycle must die first: holding a
            -- rotation now would twist the falling piece.
            release_all(); hold_btn = nil; hold_dir = nil; rel_at = nil;
          end
          plan = { steps = m.steps, idx = 1,
                   next_frame = frame + 1, born_frame = frame };
          consumed_for = m.for_frame;
          exec = "play";
          done_hold = false;
          if ORI_ENABLED then verify_armed = "spawn" end
          console.log(string.format(
            "PLAYER: plan adopted (%d steps) for f=%d",
            #m.steps, m.for_frame));
        elseif m.phase == "play" and tmr < lockval then
          -- already consumed (stale re-adoption blocked): silent — this
          -- fires every frameend of the old Down loop; the quiet is what
          -- lets quiescence form its stable pair and publish the lock.
        elseif m.phase == "play" then
          -- matched play starved by the timer gate (piece not falling?):
          -- log once per publish. No behavior change.
          local gk2 = "playgate/" .. tostring(lastpub);
          if gk2 ~= lastmis then
            lastmis = gk2;
            console.log(string.format(
              "PLAYER: play ready but tmr=%d >= lockval=%d (for_frame=%d)",
              tmr, lockval, m.for_frame));
          end
        end
      elseif m ~= nil then
        -- moves file present but not for the latest publish (stale path or
        -- Python deciding on an older board): log once per pair — this
        -- discriminates the for_frame-mismatch and path-mismatch hunches.
        local mk = string.format("mis/%d/%s/%s", m.for_frame, m.phase,
                                 tostring(lastpub));
        if mk ~= lastmis then
          lastmis = mk;
          console.log(string.format(
            "PLAYER: moves waiting (for_frame=%d lastpub=%s phase=%s "
            .. "tmr=%d lockval=%d)",
            m.for_frame, tostring(lastpub), m.phase, tmr, lockval));
        end
      elseif frame - lastnomoves >= 600 then
        -- no moves file at all (mirror writing elsewhere?): throttled.
        lastnomoves = frame;
        console.log("PLAYER: no moves file at " .. CONFIG.MOVES
                    .. " (mirror writing elsewhere?)");
      end
      if done_hold and hold_btn == nil then hold_release(); end
    end

    -- 5. publishes. Spawn reuses the cached lock board (the falling piece
    -- must never pollute the model); lock boards come from quiescence.
    -- Timer is compared against the live lock-delay value so level-ups
    -- (lock 30 -> 26 -> ...) keep working.
    if turnover then
      -- Backstop: if quiescence never published this cycle (500+ ARE can
      -- be shorter than the stable-pair window), refresh the cache with
      -- one settled scan so the board can't freeze. lock_watch still
      -- armed here means no publish happened (publish/cap both clear it).
      -- Deliberately SILENT (no publish): a post-turnover lock publish
      -- would trigger a hold move for a spawn that already sampled —
      -- useless for IRS and adoptable mid-fall, twisting the new piece.
      -- The spawn below reuses the fresh cache as usual.
      if cnt < lastcnt then board_cache = nil end  -- new game
      if lock_watch and board_cache ~= nil then
        local b_rows = scan_board();
        if b_rows ~= nil and board_settled(b_rows) then
          board_cache = b_rows;
        end
      end
      lock_watch = false;
      local rows = board_cache;
      if rows == nil then
        rows = scan_board();
        if rows == nil then
          console.log("PLAYER ERROR: spawn scan failed");
        end
      end
      if rows ~= nil then
        publish_rows(rows, cnt, cur, nxt, tmr, "spawn",
                     hold_dir or "-");
      end
    elseif tmr == lockval and prevtmr ~= nil and prevtmr < lockval then
      -- lock edge: arm quiescence; first scan lands after palette
      -- conversion (an immediate scan would dedupe-suppress the lock).
      lock_watch = true; lock_frame = frame; next_qscan = frame + 2;
      last_qkey = nil;
      -- the falling piece just locked: any remaining plan steps belong to
      -- thin air now. Stop them here so they can't dribble into ARE
      -- (eaten inputs + DAS pre-charge looked like "random moves").
      if plan ~= nil then
        console.log("PLAYER: locked, stopping plan");
        plan = nil; done_hold = true; exec = "done";
        release_all();
      end
    end
    -- 6. quiescence scans while the lock settles (covers lock flash and
    -- line-clear delays). Scans run every 2f so a stable pair fits even
    -- in short 500+ AREs. Publish only on a STABLE PAIR (two consecutive
    -- identical scans): during conversion the board flickers, and every
    -- flicker would otherwise publish a different board and flip-flop
    -- Python's decision mid-piece (diag: 3 publishes/lock with alternating
    -- irs/steps). Cap at +24f with a best-effort publish.
    -- Frame-gated only (no timer condition): Down-locked pieces never raise
    -- the timer edge, and the timer counts through clear delays anyway.
    if lock_watch and not turnover
       and frame >= next_qscan and frame <= lock_frame + 24 then
      local rows, err = scan_board();
      if rows == nil then
        console.log("PLAYER ERROR: " .. tostring(err));
      else
        local key = board_key(rows, cnt, cur, nxt);
        local settled = board_settled(rows);
        if key == last_qkey and key ~= lastkey and settled then
          local f = publish_rows(rows, cnt, cur, nxt, tmr, "lock", "-");
          if f ~= nil then board_cache = rows end
          lock_watch = false;
        elseif frame > lock_frame + 20 then
          -- cap failsafe: publish (and cache) only settled boards; a
          -- persistently transient board is skipped (safe: natural lock).
          if settled then
            local f = publish_rows(rows, cnt, cur, nxt, tmr, "lock", "-");
            if f ~= nil then board_cache = rows end
          else
            console.log("PLAYER: unsettled at cap, skipping lock publish");
          end
          lock_watch = false;
        end
        last_qkey = key;
      end
      next_qscan = frame + 2;
    elseif lock_watch and (turnover or frame > lock_frame + 24) then
      lock_watch = false;
    end
    lastcnt, lastcur, lastnxt, prevtmr = cnt, cur, nxt, tmr;
  end);
  if not ok then console.log("PLAYER ERROR: " .. tostring(err)) end
end, "tgm_player");
console.log("PLAYER: armed (" .. CONFIG.STEP_FRAMES ..
            "f/step). Publish on spawn/lock; plays " .. CONFIG.MOVES .. ".");
