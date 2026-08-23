# Pestivid colour tokens

Every token below names the physical thing it was sampled from. The research finding that forced
this rule: *"a tell is an unspecified default, not a banned colour — colour-for-colour
substitution is a non-fix."* If a token has no provenance line, it is a default and it will read
as generated regardless of hue.

## Why the previous palette failed, in numbers

`#7d8b52` `#8b9a5c` `#91a05f` `#94a364` — the four "different" greens — span **0.6° of OKLCH hue**
and **0.009 chroma**. Only lightness varied. That is the "accent used at exactly one saturation
everywhere" tell, measured.

They also sat in a dead zone: chroma 0.081–0.090, where published specs put neutrals at
0.005–0.015 and accents at 0.12–0.22. Five times too chromatic to work as a neutral, ~30% too
dull to work as an accent. That is the mechanical reason the screens read flat.

And Farmers Business Network has **deprecated** this exact family — their legacy tokens are
literally prefixed `--deprecated-color-`, and they are the closest match in a 21-product sample
to the hexes above. The desaturated sage family is the palette a real agricultural company
retired, not one it chose.

## Grounds and ink

| token | value | sampled from |
|---|---|---|
| `--ground` | `#f6f3ed` | jai-kisan.com live stylesheet, page ground |
| `--surface` | `#fffefb` | one step up from ground, warm-tinted, never pure white |
| `--surface-alt` | `#f1ede4` | zebra band, one step down from ground |
| `--ink` | `#0b1c1d` | jai-kisan.com `--primary-green-0`, used as body ink |
| `--ink-2` | `#55605c` | tinted secondary — a *green-black* grey, not a neutral grey |
| `--ink-3` | `#847f7b` | warm tertiary, chroma 0.009 so it reads neutral |
| `--rule` | `#dcd6c9` | hairline, warm |

`--ink` is the load-bearing choice. A near-black green as body text makes every page feel
agricultural **without a single green surface**, which is exactly the problem we had.

## Action and status

| token | value | pairs with | sampled from |
|---|---|---|---|
| `--action` | `#f3a537` | `#000000` text | climate.com FieldView button fill, with black label |
| `--action-ink` | `#000000` | — | FieldView uses black on amber, not white |
| `--alarm` | `#a71930` | `#ffffff` | caseih.com `--ext--color--primary` |
| `--attention` | `#8a6d3b` on `#fcf8e3` | — | enam.gov.in (Govt of India mandi portal) warning pair |
| `--proved` | `#294793` | `#ffffff` | indigo dye, still in agricultural use |
| `--healthy` | `#1a8e46` | `#ffffff` | jai-kisan.com `--primary-green-04` |

`--healthy` is the **only** green permitted in UI chrome, on exactly one thing: a healthy-leaf
verdict. Two-thirds of the 21 real agricultural products sampled do not use green as their action
colour at all; amber and blue are each more common.

## Crop and field imagery

Canvas-sampled pixel averages of **real** Indian agricultural hero photography
(samunnati.com `home-hero-bg-mob.jpg`, ninjacart.com `puttheirfaith_*`):

| token | value | what it is |
|---|---|---|
| `--sky` | `#57626d` | blue-grey — the largest area in a real field photo |
| `--soil` | `#4f463c` | turned earth |
| `--earth` | `#7f5335` | warm dry ground |
| `--crop-gold` | `#d3b663` | ripening crop |
| `--crop-dark` | `#2a3024` | crop foliage in shadow, L 20% |
| `--sky-pale` | `#e2e4e9` | overexposed sky |

Real Indian agricultural photography **does not average green.** The previous olives at L 43–52%
were lighter and greyer than any crop shot measured. Where foliage must read green it goes to
`--crop-dark` at L 20%, not a mid-tone olive.

## Coverage gate

Accent-coloured pixels stay **under 5% of any frame**. The 60% dominant is `--ground` or `--ink`,
never a hue. Check: screenshot a surface, count accent pixels.

## Build gate

Fail the build if any token has OKLCH hue 100–165 with chroma > 0.02, except `--healthy` and the
`--crop-*` imagery tokens. Written guidance does not hold — one designer found that banning
indigo pushed the model to emerald across 41 overnight builds, and only a mechanical rule stopped
it.

## Hue families

Eight, so no single one dominates: warm neutral, near-black green (ink), amber, oxblood, indigo,
sky blue-grey, soil brown, crop gold. Green is one family of eight, and in chrome it is one token.
The structural precedent is Jai Kisan's shipping 61-token system spanning ten hue families.
