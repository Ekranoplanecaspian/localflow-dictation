import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import FlowBar from "./FlowBar";
import "./styles/tokens.css";

// One bundle, two windows: the flow bar is the same app under a hash, which keeps the build a
// single entry point and lets both windows share the design tokens.
const isFlowBar = window.location.hash.startsWith("#flowbar");

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>{isFlowBar ? <FlowBar /> : <App />}</React.StrictMode>,
);
