import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import FlowBar from "./FlowBar";
import { ErrorBoundary, logUnhandledErrors } from "./hub/ErrorBoundary";
import "./styles/tokens.css";

// One bundle, two windows: the flow bar is the same app under a hash, which keeps the build a
// single entry point and lets both windows share the design tokens.
const isFlowBar = window.location.hash.startsWith("#flowbar");
document.documentElement.classList.toggle("flowbar", isFlowBar); // scopes FlowBar.css's page rules
logUnhandledErrors(isFlowBar ? "flow bar" : "hub");

// Each page has its own boundary (App.tsx); this one catches what is left - the frame itself,
// onboarding, the flow bar - so a fault never leaves a blank window.
ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ErrorBoundary page={isFlowBar ? "flow bar" : "window"} scope="window">
      {isFlowBar ? <FlowBar /> : <App />}
    </ErrorBoundary>
  </React.StrictMode>,
);
