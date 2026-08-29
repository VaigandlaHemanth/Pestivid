# What four shipped products actually measure

Captured 2026-08-30 at 1440x900 with the `taste` skill's own `extract.js`, run
against `linear.app`, `stripe.com`, `notion.so` and `vercel.com`, and then
against our own `profile.html` with the identical extractor so the numbers sit
in one table. Nothing here is an impression. Every figure is a count of
computed styles on a rendered page.

## The table that decided the work

| | text sizes (top 5, by count) | weights | spacing units |
|---|---|---|---|
| **Pestivid, before** | 17px(13) 15px(9) 12.5px(4) 27px(2) | 400(18) **700(10) 600(8)** | 12px(15) 15px(12) 14px(7) 2px(7) |
| linear.app | 12px(187) 14px(180) 13px(134) 15px(65) | 400(469) 510(138) 500(14) | **8px(138)** 4px(110) 6px(65) 12px(44) |
| stripe.com | 16px(190) 10px(93) 14px(32) 26px(24) | **300(294) 400(146)** | **8px(53)** 6px(38) 16px(31) 32px(24) |
| notion.so | 14px(67) 16px(16) 20px(4) 12px(4) | 500(59) 400(23) 700(16) | **8px(57)** 3px(46) 6px(42) 24px(29) |
| vercel.com | 14px(128) 12px(12) 11px(10) 24px(8) | 400(113) 500(41) 450(16) | 2px(340) 6px(95) 12px(30) |

Three findings, in order of how much they mattered.

**1. Half our text was bold.** 18 of 36 sampled elements were weight 600 or 700.
Stripe ships **nothing above 400** — its body is 300, one step LIGHTER than
regular. Vercel ships nothing above 500. Linear's bold is about two per cent of
its text. When everything is bold nothing recedes, which is the whole reason a
back link kept "standing out" from every position it was tried in: it was not
badly placed, it was 14.5px at weight 600 in the loudest blue in the palette,
competing with a heading that was also bold. This is the single highest-leverage
number on the page.

**2. Our body size was larger than any of theirs.** A 17px row label is bigger
than every one of these products' BODY text. Larger type is a deliberate choice
for a farmer on a phone in sunlight -- but 17px AND weight 600 AND a 27px bold
heading above it is three loud things at once, and the page has a "Bigger text"
switch for the people who need one.

**3. Our spacing was on no grid.** 12, 15, 14, 2. The three light references all
hammer **8px** as their most-used unit.

## Two things worth NOT copying

- **Shadows.** Notion's are `rgba(0,0,0,0.01)` and Vercel's are literally
  `rgba(0,0,0,0)` -- transparent. Separation on all four comes from hairlines
  and space, not elevation. That matches our own paper-square rule, so nothing
  changed. It does confirm the direction: cards were the wrong instrument.
- **Motion.** Their transitions run 0.1s-0.3s (linear 0.1/0.16, notion 0.15/0.2,
  vercel 0.15/0.3, stripe 0.24). Ours are `--t-smooth 746ms` and
  `--t-snappy 568ms` -- two to five times slower than all four. Left alone for
  now because the motion tokens are pre-solved springs and changing them is a
  product-wide decision, but it is the next number worth arguing about.

Also measured: none of the four caps its body width (`containerMaxWidth: none`
on all four at a 1440px container). That is evidence for the full-bleed layout
this product already uses, and against the width cap that was tried and rejected.

## What was changed on profile.html because of this

Row label 17px/600 -> 16px/500. Its second line 15px -> 14px. Section labels
12.5px/700 -> 12px/500. The disclosure headline 15.5px/700 -> 15px/600. Row
padding 15px -> 16px and gap 14px -> 16px, onto the 8px grid. Language chips
17px -> 15px.

Weight 600-or-heavier fell from 50% of sampled text to the heading, the avatar
initial, and the app bar -- which is what those weights are for.

**One regression this caused, and the fix.** Shrinking the chips from
`12px 15px` padding at 17px to `10px 16px` at 15px took them to 41px tall,
under the 44px minimum, on the one control a farmer who cannot read the screen
needs most. The type stayed at the measured size and the BOX kept its target:
`min-height: 44px` with inline-flex centring. Measured back at 79x44.

## The remaining heavy text, and why it stays

`Pestivid`(700), the unread badge(700) and the bar avatar(600) are the app bar,
shared by all 21 pages -- not a one-page decision. `Demo Farmer`(27px/700) and
the avatar initial(700) are the page's heading, and Notion uses 700 for headings
too. Those are the only bold things left on the screen, which is the point.
