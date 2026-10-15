---
title: RailSentinel
emoji: 🚆
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: AI rail travel governed by an autonomous AI control tower
---

# RailSentinel

AI rail-travel services (confirmation prediction, recommendations, delays, an agentic planner, fares, a grounded
support assistant and fraud detection) governed in real time by an autonomous supervisor agent.

This Space runs the Docker image from the main repository: https://github.com/mimaniharsh/railsentinel

* Open the **Chaos Lab** page and run **Autopilot** on any scenario to watch the control tower detect, diagnose and
  repair a failure.
* Optional secrets: `GEMINI_API_KEY` (enables Gemini for the planner and support assistant), `RS_API_KEYS`,
  `RS_PASSPORT_SECRET`.
* Storage is ephemeral: the demo returns to its trained state when the Space restarts.

All passenger, train, fare and delay data is simulated by the platform's digital twin on the real Indian Railways
station network. Not affiliated with Indian Railways or IRCTC.
