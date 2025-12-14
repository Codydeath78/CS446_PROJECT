//This should be everything. end of semester project.
"use client";
import { Box, TextField, Stack, Button, IconButton, Tooltip, CssBaseline } from "@mui/material";
import SendRoundedIcon from "@mui/icons-material/SendRounded";
import AutoAwesomeRoundedIcon from "@mui/icons-material/AutoAwesomeRounded";
import DarkModeRoundedIcon from "@mui/icons-material/DarkModeRounded";
import LightModeRoundedIcon from "@mui/icons-material/LightModeRounded";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import { useEffect, useMemo, useRef, useState } from "react";
import Markdown from "react-markdown";
import { motion, AnimatePresence } from "framer-motion";
import ShieldRoundedIcon from "@mui/icons-material/ShieldRounded";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import WarningAmberRoundedIcon from "@mui/icons-material/WarningAmberRounded";
import { Snackbar, Alert, Slide, Popover } from "@mui/material";
import { useRouter } from "next/navigation";

//main Chat Page
export default function Home() {
  //next.js router
  const router = useRouter();

  //chat state
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: `Hi! I'm your chatbot assistant. how can I help you today?`,
      createdAt: Date.now(),
    },
  ]);
  //composer state
  const [message, setMessage] = useState("");
  //sending state
  const [isSending, setIsSending] = useState(false);
  //theme mode state
  const [mode, setMode] = useState(() => {
    if (typeof window === "undefined") return "light";
    return localStorage.getItem("chat-ui-mode") || "light";
  });

  //toast state
  const [toast, setToast] = useState({ open: false, msg: "" });
  //compare popover state
  const [compareAnchor, setCompareAnchor] = useState(null);
  //compare data state
  const [compareData, setCompareData] = useState({ before: "", after: "" });
  //compare tab state
  const [compareTab, setCompareTab] = useState("after"); // "before" | "after"

  // dynamically import diff library
  let _DiffLib = null;
  async function getDiffLib() {
    if (_DiffLib) return _DiffLib;
    _DiffLib = await import("diff");
    return _DiffLib;
  }
  //construct diff parts for before/after comparison
  //build “parts” = [{added?, removed?, value}]
  async function buildDiffParts(before, after) {
    const Diff = await getDiffLib();
    //word diff that keeps spaces means better phrase readability
    const parts = Diff.diffWordsWithSpace(before || "", after || "", { ignoreCase: false });
    return parts;
  }

  //open compare popover
  function openCompare(e, before, after) {
    setCompareAnchor({ top: e.clientY, left: e.clientX });
    setCompareData({ before, after });
    setCompareTab("before");
    buildDiffParts(before, after).then((parts) => {
    setCompareData((d) => ({ ...d, parts }));
    }).catch(() => {});
  }

  //close compare popover
  function closeCompare() {
    setCompareAnchor(null);
  }


  //show toast message
  function showToast(msg) {
    setToast({ open: true, msg });
    //hard-close after 1.2s even if the Snackbar’s timer is paused
    setTimeout(() => setToast({ open: false, msg: "" }), 1200);
  }
  //snackbar transition
  function TransitionUp(props) {
    return <Slide {...props} direction="up" />;
  }
  //scroll ref
  const scrollRef = useRef(null);
  //ref to track if receive sound played
  const playedReceiveRef = useRef(false);

  //play sound effect
  const playSound = (type) => {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.connect(g); g.connect(ctx.destination);
      const now = ctx.currentTime;
      const isSend = type === "send";
      o.type = "sine";
      o.frequency.setValueAtTime(isSend ? 760 : 520, now);
      g.gain.setValueAtTime(0.0001, now);
      g.gain.exponentialRampToValueAtTime(0.2, now + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, now + 0.14);
      o.start(now);
      o.stop(now + 0.16);
    } catch {}
  };

  //auto-scroll to the latest message
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [messages.length]);

  // Persist theme
  useEffect(() => {
    if (typeof window !== "undefined") localStorage.setItem("chat-ui-mode", mode);
  }, [mode]);

  //create MUI theme
  const theme = useMemo(() => createTheme({
    palette: {
      mode,
      primary: { main: mode === "light" ? "#6366f1" : "#8b95ff" },
      background: {
        default: mode === "light" ? "#eef2ff" : "#0b1220",
        paper: mode === "light" ? "#ffffff" : "#0f172a",
      },
    },
    shape: { borderRadius: 16 },
  }), [mode]);

  //send message handler
  const sendMessage = async () => {
  if (!message.trim()) return;

  const userMsg = { role: "user", content: message, createdAt: Date.now() };
  setMessage("");
  setIsSending(true);
  playedReceiveRef.current = false;

  //append only the user message first
  setMessages((m) => [...m, userMsg]);
  playSound("send");

  //call the API to get assistant response stream
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify([...messages, userMsg]),
    });

    if (!res.body) throw new Error("No response body");

    //read moderation headers and decide if we should delay showing assistant
    const action = (res.headers.get("x-moderation-action") || "").toLowerCase();
    const safety = res.headers.get("x-safety"); // null | "user" | "assistant"
    let safetyScores = null;
    try { safetyScores = JSON.parse(res.headers.get("x-safety-scores") || "null"); } catch {}
    const safetyReason = res.headers.get("x-safety-reason") || null;

    //original text before rephrase
    const safetyOriginal = res.headers.get("x-safety-original") || null;

    let delayMs = 0;
    if (action.startsWith("rephrased-user")) {
      delayMs = 1200;
      showToast("Moderator detected biased phrasing — rewriting your message…");
    } else if (action.startsWith("rephrased-assistant")) {
      delayMs = 1200;
      showToast("Moderator detected biased phrasing — rewriting assistant draft…");
    } else if (action.startsWith("blocked")) {
      delayMs = 1200;
      showToast("Moderator blocked unsafe content.");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    //stream, but buffer until the delay is over, then create the assistant bubble once
    let buffer = "";
    let assistantShown = false;
    const unblockAt = Date.now() + delayMs;

    //read the streaming chunks
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;

      //if it's time to reveal the assistant bubble
      if (!assistantShown && Date.now() >= unblockAt) {
        setMessages((msgs) => [
          ...msgs,
          {
            role: "assistant",
            content: buffer,
            createdAt: Date.now(),
            //attach safety flags on creation so tooltip works
            ...(safety ? { safety, safetyScores, safetyReason, safetyOriginal } : {}),
          },
        ]);
        assistantShown = true;
        // play receive sound
        if (!playedReceiveRef.current) {
          playedReceiveRef.current = true;
          playSound("receive");
        }
      } else if (assistantShown) {
        //update the last assistant bubble as more chunks arrive
        setMessages((msgs) => {
          const last = msgs[msgs.length - 1];
          const rest = msgs.slice(0, -1);
          return [...rest, { ...last, content: last.content + chunk }];
        });
      }
    }

    //edge case: very short responses may finish before unblock time
    if (!assistantShown) {
      //wait out any remaining delay
      const remain = unblockAt - Date.now();
      if (remain > 0) await new Promise((r) => setTimeout(r, remain));
      //finally show the assistant message
      setMessages((msgs) => [
        ...msgs,
        {
          role: "assistant",
          content: buffer,
          createdAt: Date.now(),
          ...(safety ? { safety, safetyScores, safetyReason, safetyOriginal } : {}),
        },
      ]);
      // play receive sound
      if (!playedReceiveRef.current) {
        playedReceiveRef.current = true;
        playSound("receive");
      }
    }
  } catch (e) {
    setMessages((m) => [
      ...m,
      { role: "assistant", content: "Oops—I hit a snag while replying. Try again?", createdAt: Date.now() },
    ]);
  } finally {
    setIsSending(false);
  }
};
  //animation variants for message bubbles
  const bubbleVariants = {
    initialLeft: { opacity: 0, x: -16, filter: "blur(4px)" },
    initialRight: { opacity: 0, x: 16, filter: "blur(4px)" },
    animate: { opacity: 1, x: 0, filter: "blur(0px)", transition: { type: "spring", stiffness: 380, damping: 28 } },
    exit: { opacity: 0, y: 12, scale: 0.98, transition: { duration: 0.18 } },
  };
  //format timestamp to HH:MM
  const formatTime = (ts) => {
    const d = new Date(ts);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    return `${hh}:${mm}`;
  };
  //typing dots component
  const TypingDots = () => (
    <Box sx={{ display: "flex", gap: 0.8, alignItems: "center", py: 0.5 }}>
      {[0, 1, 2].map((i) => (
        <Box
          key={i}
          component={motion.span}
          sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: "primary.light" }}
          animate={{ y: [0, -4, 0], opacity: [0.5, 1, 0.5] }}
          transition={{ repeat: Infinity, duration: 1.2, delay: i * 0.15 }}
        />
      ))}
    </Box>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{
          width: "100vw",
          height: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          p: { xs: 1.5, sm: 3 },
          background: mode === "light"
            ? "radial-gradient(1200px 600px at 10% 10%, rgba(99,102,241,0.25), transparent), radial-gradient(1000px 500px at 90% 30%, rgba(56,189,248,0.25), transparent), radial-gradient(800px 400px at 50% 90%, rgba(236,72,153,0.22), transparent)"
            : "radial-gradient(1200px 600px at 10% 10%, rgba(99,102,241,0.12), transparent), radial-gradient(1000px 500px at 90% 30%, rgba(56,189,248,0.12), transparent), radial-gradient(800px 400px at 50% 90%, rgba(236,72,153,0.1), transparent)",
        }}
      >
        <Box
          component={motion.div}
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          sx={{
            width: { xs: "100%", sm: 520, md: 640 },
            height: { xs: "96vh", sm: "88vh" },
            display: "flex",
            flexDirection: "column",
            borderRadius: 4,
            overflow: "hidden",
            position: "relative",
            boxShadow: mode === "light" ? "0 20px 50px rgba(2, 6, 23, 0.35)" : "0 20px 50px rgba(0,0,0,0.6)",
            backdropFilter: "blur(14px)",
            border: "1px solid rgba(255,255,255,0.18)",
            background: theme.palette.background.paper,
          }}
        >
          {/* header */}
          <Box
            sx={{
              px: 2.5,
              py: 1.75,
              display: "flex",
              alignItems: "center",
              gap: 1.25,
              borderBottom: "1px solid rgba(0,0,0,0.06)",
              background: mode === "light" ? "linear-gradient(90deg, rgba(99,102,241,0.15), rgba(56,189,248,0.12))" : "linear-gradient(90deg, rgba(99,102,241,0.08), rgba(56,189,248,0.08))",
            }}
          >
            <Box component={motion.div} initial={{ rotate: -12, scale: 0.8 }} animate={{ rotate: 0, scale: 1 }} transition={{ type: "spring", stiffness: 260, damping: 14 }}>
              <AutoAwesomeRoundedIcon color="primary" />
            </Box>
            <Box component={motion.h3} style={{ margin: 0, fontWeight: 700, fontSize: 16 }}>Assistant</Box>
            <Box sx={{ ml: "auto", display: "flex", alignItems: "center", gap: 0.5 }}>
              <Tooltip title={mode === "light" ? "Switch to dark" : "Switch to light"}>
                <IconButton onClick={() => setMode((m) => (m === "light" ? "dark" : "light"))}>
                  {mode === "light" ? <DarkModeRoundedIcon /> : <LightModeRoundedIcon />}
                </IconButton>
              </Tooltip>

              {/*button to open review UI */}
              <Tooltip title="Open reviewer UI">
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => router.push("/review")}
                  sx={{
                    textTransform: "none",
                    ml: 0.5,
                    borderRadius: 999,
                    px: 1.5,
                    py: 0.25,
                    fontSize: 12,
                  }}
                >
                  Review
                </Button>
              </Tooltip>
            
            </Box>
          </Box>

          {/* messages */}
          <Stack ref={scrollRef} direction="column" spacing={1.25} sx={{ flexGrow: 1, overflow: "auto", p: 2, scrollBehavior: "smooth" }}>
            <AnimatePresence initial={false}>
              {messages.map((m, i) => {
                const isAssistant = m.role === "assistant";
                const showTyping = isAssistant && m.content === "";
                return (
                  <Box key={i} sx={{ display: "flex", justifyContent: isAssistant ? "flex-start" : "flex-end" }}>
                    <Box
                      component={motion.div}
                      initial={isAssistant ? "initialLeft" : "initialRight"}
                      animate="animate"
                      exit="exit"
                      variants={bubbleVariants}
                      whileHover={{ scale: 1.01 }}
                      style={{ maxWidth: "75%" }}
                    >
                      <Box
                        sx={{
                          px: 2,
                          py: 1.5,
                          borderRadius: 3,
                          color: isAssistant ? (mode === "light" ? "#0b1220" : "#e5e7eb") : "white",
                          bgcolor: isAssistant ? (mode === "light" ? "rgba(255,255,255,0.9)" : "#111827") : "primary.main",
                          boxShadow: isAssistant ? (mode === "light" ? "0 8px 24px rgba(2, 6, 23, 0.08)" : "0 8px 24px rgba(0,0,0,0.5)") : "0 10px 30px rgba(99,102,241,0.35)",
                          border: isAssistant ? "1px solid rgba(0,0,0,0.06)" : "none",
                          backdropFilter: isAssistant ? "blur(6px)" : undefined,
                        }}
                      >
                        {showTyping ? (
                          <TypingDots />
                        ) : (
                          <>
                            <Markdown>{m.content}</Markdown>

                            {/*safety badge (only when rephrased) */}
                            {m.safety && (
                              <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mt: 0.75 }}>
                                {/* Shield icon (visual) */}
                                <ShieldRoundedIcon
                                  fontSize="small"
                                  sx={{ opacity: 0.9 }}
                                  aria-label="Safety moderation applied"
                                />

                                {/*info icon so Detector scores */}
                                <Tooltip
                                  title={
                                    <Box sx={{ maxWidth: 280 }}>
                                      <Box sx={{ fontWeight: 700, mb: 0.5 }}>Detector scores</Box>
                                      <Box sx={{ fontSize: 12, opacity: 0.9 }}>
                                        {m.safetyScores
                                          ? (
                                            <>
                                              biased: {Number(m.safetyScores.biased).toFixed(3)} | neutral: {Number(m.safetyScores.neutral).toFixed(3)}
{/* small confidence meter */}
<Box
  sx={{
    mt: 1,
    width: 60,
    height: 6,
    display: "flex",
    borderRadius: 999,
    overflow: "hidden",
    border: "1px solid rgba(0,0,0,0.2)",
  }}
>
  <Box
    sx={{
      width: `${Math.round(Number(m.safetyScores.biased) * 100)}%`,
      bgcolor: "rgba(236,72,153,0.9)", //pink/red biased side
      transition: "width 0.3s",
    }}
  />
  <Box
    sx={{
      flexGrow: 1,
      bgcolor: "rgba(56,189,248,0.35)", //cyan neutral side
    }}
  />
</Box>
  </>
    )
     : "Scores unavailable."
      }
       </Box>
         </Box>
           }
             arrow
               >
                 <InfoOutlinedIcon
                    fontSize="small"
                      sx={{ opacity: 0.85, cursor: "help" }}
                        aria-label="Detector scores"
                          />
                            </Tooltip>
{/* warning icon so reason and original (before rephrase) */}
    <Tooltip
      title={
        <Box sx={{ maxWidth: 320 }}>
          <Box sx={{ fontWeight: 700, mb: 0.5 }}>
            {m.safety === "user" ? "Why your message was rephrased" : "Why assistant reply was rephrased"}
          </Box>
          <Box sx={{ fontSize: 12, opacity: 0.9, mb: 0.75 }}>
            {m.safetyReason || "Flagged by moderator to keep the conversation safe."}
          </Box>
          {m.safetyOriginal && (
            <>
              <Box sx={{ fontWeight: 700, mb: 0.25, fontSize: 12 }}>Original (before rephrase)</Box>
              <Box
                component="pre"
                sx={{
                  m: 0,
                  p: 1,
                  fontSize: 12,
                  lineHeight: 1.35,
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  maxHeight: 180,
                  overflow: "auto",
                  bgcolor: "rgba(0,0,0,0.06)",
                  borderRadius: 1,
                }}
              >
                {m.safetyOriginal}
              </Box>

              {/* compare (Before / After) link */}
          <Box sx={{ mt: 0.75, textAlign: "right" }}>
            <Button
              size="small"
              variant="text"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                openCompare(e, m.safetyOriginal || "", m.content || "");
              }}
              sx={{ textTransform: "none", fontSize: 12, px: 0.5, minWidth: "auto" }}
            >
              Compare (Before / After)
            </Button>
          </Box>
            </>
          )}
        </Box>
      }
      arrow
    >
      <WarningAmberRoundedIcon
        fontSize="small"
        sx={{ opacity: 0.9, cursor: "help" }}
        aria-label="Reason & original text"
      />
    </Tooltip>

    <Box sx={{ fontSize: 12, opacity: 0.7 }}>
      Rephrased for safety
        </Box>
          </Box>
            )}
              <Box sx={{ mt: 0.5, fontSize: 11, opacity: 0.6, textAlign: isAssistant ? "left" : "right" }}>
                 {formatTime(m.createdAt)}
                    </Box>
                      </>
                      )}
                      </Box>
                    </Box>
                  </Box>
                );
              })}
            </AnimatePresence>
          </Stack>

          {/* composer */}
          <Box sx={{ p: 2, pt: 1.25, borderTop: "1px solid rgba(0,0,0,0.06)", background: mode === "light" ? "linear-gradient(180deg, rgba(255,255,255,0.65), rgba(255,255,255,0.45))" : "linear-gradient(180deg, rgba(17,24,39,0.85), rgba(17,24,39,0.7))" }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <TextField
                label="Message"
                placeholder="Type a message…"
                fullWidth
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                variant="outlined"
                size="medium"
                sx={{
                  "& .MuiOutlinedInput-root": {
                    borderRadius: 3,
                    transition: "transform 240ms ease, box-shadow 240ms ease",
                    backdropFilter: "blur(6px)",
                  },
                  "& .MuiOutlinedInput-root:hover": {
                    boxShadow: mode === "light" ? "0 12px 30px rgba(2,6,23,0.1)" : "0 12px 30px rgba(0,0,0,0.45)",
                  },
                  "& .MuiOutlinedInput-root.Mui-focused": {
                    transform: "translateY(-1px)",
                    boxShadow: mode === "light" ? "0 18px 40px rgba(56,189,248,0.25)" : "0 18px 40px rgba(99,102,241,0.35)",
                  },
                }}
              />
              <Button
                onClick={sendMessage}
                disabled={isSending}
                variant="contained"
                endIcon={<SendRoundedIcon />}
                sx={{
                  borderRadius: 999,
                  px: 2.25,
                  py: 1,
                  textTransform: "none",
                  fontWeight: 700,
                  boxShadow: mode === "light" ? "0 12px 30px rgba(99,102,241,0.35)" : "0 12px 30px rgba(99,102,241,0.55)",
                  transition: "transform 160ms ease, filter 160ms ease",
                  "&:hover": { transform: "translateY(-2px)" },
                  "&:active": { transform: "translateY(0px) scale(0.98)" },
                }}
              >
                {isSending ? "Sending…" : "Send"}
              </Button>
            </Stack>
          </Box>
        </Box>
      </Box>
      <Snackbar
        open={toast.open}
        slots={{ transition: TransitionUp }}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
        autoHideDuration={1200}
        onClose={() => setToast({ open: false, msg: "" })}
        sx={{ zIndex: 9999 }}
      >
        <Alert
          elevation={0}
          severity="info"
          sx={{
            borderRadius: 999,
            px: 2,
            py: 0.5,
            fontSize: 13,
            backdropFilter: "blur(8px)",
            bgcolor: mode === "light" ? "rgba(99,102,241,0.12)" : "rgba(99,102,241,0.2)",
            border: "1px solid rgba(99,102,241,0.35)"
          }}
        >
          {toast.msg}
        </Alert>
      </Snackbar>

      <Popover
  open={Boolean(compareAnchor)}
  anchorEl={null}
  anchorReference="anchorPosition"
  anchorPosition={compareAnchor || { top: 0, left: 0 }}
  onClose={closeCompare}
  anchorOrigin={{ vertical: "top", horizontal: "center" }}
  transformOrigin={{ vertical: "bottom", horizontal: "center" }}
  slotProps={{
    paper: {
      sx: {
        p: 1.5,
        maxWidth: 420,
        width: 420,
        borderRadius: 2,
        boxShadow: mode === "light" ? 4 : 8,
        bgcolor: mode === "light" ? "#fff" : "#0f172a",
        border: "1px solid",
        borderColor: mode === "light" ? "rgba(0,0,0,0.08)" : "rgba(255,255,255,0.12)",
      },
    },
  }}
>
  <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1 }}>
    <Box sx={{ fontWeight: 700 }}>Compare</Box>
    <Box sx={{ display: "flex", gap: 1 }}>
      <Button
        size="small"
        variant={compareTab === "before" ? "contained" : "text"}
        onClick={() => setCompareTab("before")}
        sx={{ textTransform: "none", minWidth: 64 }}
      >
        Before
      </Button>
      <Button
        size="small"
        variant={compareTab === "after" ? "contained" : "text"}
        onClick={() => setCompareTab("after")}
        sx={{ textTransform: "none", minWidth: 64 }}
      >
        After
      </Button>
    </Box>
  </Box>

  <Box
    component="pre"
    sx={{
      m: 0,
      p: 1,
      borderRadius: 1.5,
      whiteSpace: "pre-wrap",
      wordBreak: "break-word",
      maxHeight: 320,
      overflow: "auto",
      bgcolor: mode === "light" ? "rgba(0,0,0,0.04)" : "rgba(255,255,255,0.06)",
      border: "1px solid",
      borderColor: mode === "light" ? "rgba(0,0,0,0.08)" : "rgba(255,255,255,0.12)",
      fontSize: 13.5,
    }}
  >
       {Array.isArray(compareData.parts)
     ? compareData.parts.map((p, idx) => {
         
         if (compareTab === "after" && p.added) {
           return (
             <span
               key={idx}
               style={{
                 background: "rgba(56,189,248,0.28)", //cyan tint
                 borderRadius: 4,
                 padding: "0 2px",
               }}
             >
               {p.value}
             </span>
           );
         }
         
         if (compareTab === "before" && p.removed) {
           return (
             <span
               key={idx}
               style={{
                 background: "rgba(236,72,153,0.28)", //pink/red tint
                 borderRadius: 4,
                 textDecoration: "line-through",
                 padding: "0 2px",
               }}
             >
               {p.value}
             </span>
           );
         }
         
         return <span key={idx}>{p.value}</span>;
       })
     : (compareTab === "before" ? compareData.before : compareData.after)}

  </Box>

  {/* only in Compare popover and only on After */}
  {compareTab === "after" && (
    <Box
      sx={{
        mt: 0.75,
        fontSize: 11.5,
        color: "text.secondary",
        opacity: 0.85,
        fontStyle: "italic",
      }}
    >
      (softened wording applied)
    </Box>
  )}
</Popover>

    </ThemeProvider>
  );
}