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

export default function Home() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: `Hi! I'm your chatbot assistant. how can I help you today?`,
      createdAt: Date.now(),
    },
  ]);
  const [message, setMessage] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [mode, setMode] = useState(() => {
    if (typeof window === "undefined") return "light";
    return localStorage.getItem("chat-ui-mode") || "light";
  });

  const scrollRef = useRef(null);
  const playedReceiveRef = useRef(false);

  // --- Sound effects (no external files; Web Audio API beeps) ---
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

  // Auto-scroll to the latest message
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [messages.length]);

  // Persist theme
  useEffect(() => {
    if (typeof window !== "undefined") localStorage.setItem("chat-ui-mode", mode);
  }, [mode]);

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

  const sendMessage = async () => {
    if (!message.trim()) return;
    const userMsg = { role: "user", content: message, createdAt: Date.now() };
    setMessage("");
    setIsSending(true);
    playedReceiveRef.current = false;
    setMessages((m) => [...m, userMsg, { role: "assistant", content: "", createdAt: Date.now() }]);
    playSound("send");

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify([...messages, userMsg]),
      });

      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        setMessages((msgs) => {
          const last = msgs[msgs.length - 1];
          const rest = msgs.slice(0, msgs.length - 1);
          // play receive sound once when first chunk lands
          if (!playedReceiveRef.current) {
            playedReceiveRef.current = true;
            playSound("receive");
          }
          return [...rest, { ...last, content: last.content + chunk }];
        });
      }
    } catch (e) {
      setMessages((m) => [
        ...m.slice(0, -1),
        { role: "assistant", content: "Oops—I hit a snag while replying. Try again?", createdAt: Date.now() },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  const bubbleVariants = {
    initialLeft: { opacity: 0, x: -16, filter: "blur(4px)" },
    initialRight: { opacity: 0, x: 16, filter: "blur(4px)" },
    animate: { opacity: 1, x: 0, filter: "blur(0px)", transition: { type: "spring", stiffness: 380, damping: 28 } },
    exit: { opacity: 0, y: 12, scale: 0.98, transition: { duration: 0.18 } },
  };

  const formatTime = (ts) => {
    const d = new Date(ts);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    return `${hh}:${mm}`;
  };

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
          {/* Header */}
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
            </Box>
          </Box>

          {/* Messages */}
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

          {/* Composer */}
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
    </ThemeProvider>
  );
}