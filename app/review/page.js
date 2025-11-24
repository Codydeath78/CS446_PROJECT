"use client";
import { useEffect, useState } from "react";
import {
  Box,
  Button,
  TextField,
  Stack,
  Typography,
  Paper,
  MenuItem,
  CircularProgress,
} from "@mui/material";
import { motion, AnimatePresence } from "framer-motion";

//categories for labeling
const categories = ["hate", "stereotype", "harassment", "toxic", "other"];

//styles for tab buttons
const tabButtonSx = (active) => ({
  px: 1.8,
  py: 0.6,
  borderRadius: 999,
  textTransform: "none",
  fontSize: 13,
  fontWeight: active ? 700 : 500,
  letterSpacing: 0.2,
  border: "none",
  color: active ? "#0b1120" : "rgba(148,163,184,0.9)",
  backgroundColor: active ? "rgba(129,140,248,1)" : "transparent",
  transition: "all 160ms ease-out",
  "&:hover": {
    backgroundColor: active
      ? "rgba(129,140,248,1)"
      : "rgba(79,70,229,0.12)",
    transform: "translateY(-1px)",
    boxShadow: active ? "0 10px 26px rgba(79,70,229,0.55)" : "none",
  },
});

//main review page component
export default function ReviewPage() {
  const [hasAccess, setHasAccess] = useState(false);
  const [password, setPassword] = useState("");
  const [loginError, setLoginError] = useState("");

  const [loading, setLoading] = useState(false);
  const [items, setItems] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState({});
  const [notes, setNotes] = useState({});
  const [sortOrder, setSortOrder] = useState("newest"); // "newest" | "oldest" | "confidence_high" | "confidence_low"
  const [sourceFilter, setSourceFilter] = useState("all"); // "all" | "user" | "assistant"
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(false);

  //handle login
  const handleLogin = async () => {
    setLoginError("");
    //send login request
    try {
      const res = await fetch("/api/reviewer_login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        setLoginError(data.error || "Invalid password");
        return;
      }
      setHasAccess(true);
    } catch (e) {
      setLoginError("Server error");
    }
  };

  //fetch pending events
  const loadItems = async (opts = {}) => {
  setLoading(true);
  //merge options with current state
  try {
    const nextSort = opts.sortOrder ?? sortOrder;
    const nextSource = opts.sourceFilter ?? sourceFilter;
    const nextPage = opts.page ?? page;
    //build query params
    const params = new URLSearchParams({
      sort: nextSort,
      source: nextSource,
      page: String(nextPage),
    });
    //fetch data
    const res = await fetch(`/api/review_events?${params.toString()}`);

    if (!res.ok) {
      // Optional: clear items so it's obvious nothing loaded
      // setItems([]);
      console.error("[review] GET failed", await res.text());
      return;
    }
    //parse response JSON
    const data = await res.json();
    //update state with new data
    setItems(data.items || []);
    setPage(data.page ?? nextPage);
    setHasMore(
      typeof data.hasMore === "boolean"
        ? data.hasMore
        : (data.items || []).length > 0
    );
  } catch (e) {
    console.error("[review] GET exception", e);
  } finally {
    setLoading(false);
  }
};
  //load items on access grant
  useEffect(() => {
    if (hasAccess) {
      loadItems({ page: 1 });
    }
  }, [hasAccess]);

  //handle labeling an item
  const handleLabel = async (id, isUnsafe) => {
    const category = selectedCategory[id];
    const note = (notes[id] || "").trim();

    if (!category) {
      alert("Please select a category before submitting your review.");
      return;
    }
    if (!note) {
      alert("Please enter a short note before submitting your review.");
      return;
    }
    //build payload
    const payload = {
      id,
      isUnsafe,
      category,
      notes: note,
    };
    //send POST request
    try {
    const res = await fetch("/api/review_events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    
      if (!res.ok) {
        console.error("[review] POST failed", await res.text());
        return;
      }

      //animate out using AnimatePresence by removing from list
      setItems((prev) => prev.filter((it) => it.id !== id));
    } catch (e) {
      console.error(e);
    }
  };

  //login UI
  if (!hasAccess) {
    return (
      <Box
        sx={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background:
            "radial-gradient(circle at 0% 0%, #1d4ed8 0, transparent 55%), radial-gradient(circle at 100% 0%, #ec4899 0, transparent 55%), #020617",
        }}
      >
        <Paper
          component={motion.div}
          initial={{ opacity: 0, y: 18, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.35, ease: "easeOut" }}
          sx={{
            p: 3,
            minWidth: 340,
            bgcolor: "rgba(15,23,42,0.96)",
            borderRadius: 4,
            border: "1px solid rgba(148,163,184,0.5)",
            boxShadow: "0 20px 45px rgba(15,23,42,0.9)",
          }}
        >
          <Typography
            variant="h6"
            sx={{
              mb: 1,
              color: "#e5e7eb",
              fontWeight: 700,
              letterSpacing: 0.4,
            }}
          >
            Safety Reviewer Login
          </Typography>
          <Typography
            sx={{ mb: 2, fontSize: 13, color: "rgba(148,163,184,0.9)" }}
          >
            Enter the admin password to review flagged safety events.
          </Typography>
          <Stack spacing={2}>
            <TextField
              label="Admin password"
              type="password"
              fullWidth
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              InputLabelProps={{ sx: { color: "#9ca3af" } }}
              InputProps={{
                sx: {
                  color: "#e5e7eb",
                  "& .MuiOutlinedInput-notchedOutline": {
                    borderColor: "rgba(148,163,184,0.5)",
                  },
                  "&:hover .MuiOutlinedInput-notchedOutline": {
                    borderColor: "rgba(129,140,248,0.9)",
                  },
                  "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                    borderColor: "rgba(129,140,248,1)",
                  },
                },
              }}
            />
            {loginError && (
              <Typography sx={{ color: "#f87171", fontSize: 13 }}>
                {loginError}
              </Typography>
            )}
            <Button
              variant="contained"
              onClick={handleLogin}
              component={motion.button}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              sx={{
                textTransform: "none",
                fontWeight: 700,
                borderRadius: 999,
                py: 1.1,
                background:
                  "linear-gradient(90deg, #4f46e5, #22d3ee, #ec4899)",
                backgroundSize: "200% 200%",
                animation: "gradientShift 6s ease infinite",
                "@keyframes gradientShift": {
                  "0%": { backgroundPosition: "0% 50%" },
                  "50%": { backgroundPosition: "100% 50%" },
                  "100%": { backgroundPosition: "0% 50%" },
                },
              }}
            >
              Enter reviewer portal
            </Button>
          </Stack>
        </Paper>
      </Box>
    );
  }

  //main reviewer UI
  return (
    <Box
      sx={{
        minHeight: "100vh",
        p: 3,
        color: "#e5e7eb",
        background:
          "radial-gradient(circle at 0% 0%, rgba(59,130,246,0.4) 0, transparent 55%), radial-gradient(circle at 100% 0%, rgba(244,114,182,0.4) 0, transparent 55%), #020617",
      }}
    >
      <Box
        component={motion.div}
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        sx={{ mb: 2, display: "flex", alignItems: "baseline", gap: 1 }}
      >
        <Typography
          variant="h5"
          sx={{ fontWeight: 700, letterSpacing: 0.4 }}
        >
          Safety Events Reviewer
        </Typography>
        <Typography
          sx={{ fontSize: 13, color: "rgba(148,163,184,0.9)" }}
        >
          Review and label biased or unsafe model behavior.
        </Typography>
        <Typography sx={{ fontSize: 13, color: "rgba(148,163,184,0.9)" }}>
          Labels below apply to the original text, not the rephrased version.
        </Typography>

      </Box>

      {/* filters and sort */}
      <Stack
        direction="row"
        spacing={3}
        sx={{
          mb: 2.5,
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        {/* refresh */}
        <Button
          size="small"
          variant="outlined"
          onClick={() => loadItems({ page: 1 })}
          component={motion.button}
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          sx={{
            textTransform: "none",
            borderRadius: 999,
            borderColor: "rgba(148,163,184,0.7)",
            color: "rgba(226,232,240,0.9)",
            px: 1.8,
            py: 0.5,
            fontSize: 13,
          }}
        >
          Refresh
        </Button>

        {/* source tabs */}
        <Stack direction="row" spacing={1.2} alignItems="center">
          <Typography sx={{ fontSize: 14, opacity: 0.85 }}>
            Source:
          </Typography>
          {["all", "user", "assistant"].map((src) => (
            <Button
              key={src}
              size="small"
              variant="text"
              component={motion.button}
              whileHover={{ y: -1 }}
              whileTap={{ scale: 0.96 }}
              onClick={() => {
                setSourceFilter(src);
                loadItems({ sourceFilter: src, page: 1 });
              }}
              sx={tabButtonSx(sourceFilter === src)}
            >
              {src === "all"
                ? "All"
                : src === "user"
                ? "User"
                : "Assistant"}
            </Button>
          ))}
        </Stack>

        {/* sort tabs */}
        <Stack direction="row" spacing={1.2} alignItems="center">
          <Typography sx={{ fontSize: 14, opacity: 0.85 }}>
            Sort:
          </Typography>
          {[
            ["newest", "Newest"],
            ["oldest", "Oldest"],
            ["confidence_high", "Highest confidence"],
            ["confidence_low", "Lowest confidence"],
          ].map(([key, label]) => (
            <Button
              key={key}
              size="small"
              variant="text"
              component={motion.button}
              whileHover={{ y: -1 }}
              whileTap={{ scale: 0.96 }}
              onClick={() => {
                setSortOrder(key);
                loadItems({ sortOrder: key, page: 1 });
              }}
              sx={tabButtonSx(sortOrder === key)}
            >
              {label}
            </Button>
          ))}
        </Stack>
      </Stack>

      {/* pagination */}
      <Stack
        direction="row"
        spacing={2}
        alignItems="center"
        sx={{ mb: 2 }}
      >
        <Button
          size="small"
          variant="outlined"
          disabled={page <= 1 || loading}
          onClick={() => {
            const nextPage = Math.max(page - 1, 1);
            loadItems({ page: nextPage });
          }}
          component={motion.button}
          whileTap={{ scale: 0.96 }}
          sx={{
            textTransform: "none",
            borderRadius: 999,
            fontSize: 13,
            borderColor: "rgba(148,163,184,0.7)",
          }}
        >
          Previous
        </Button>
        <Button
          size="small"
          variant="outlined"
          disabled={!hasMore || loading}
          onClick={() => {
            const nextPage = page + 1;
            loadItems({ page: nextPage });
          }}
          component={motion.button}
          whileTap={{ scale: 0.96 }}
          sx={{
            textTransform: "none",
            borderRadius: 999,
            fontSize: 13,
            borderColor: "rgba(148,163,184,0.7)",
          }}
        >
          Next
        </Button>
        <Typography sx={{ fontSize: 13, opacity: 0.9 }}>
          Page{" "}
          <motion.span
            key={page}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
          >
            {page}
          </motion.span>
        </Typography>
      </Stack>

      {/* loading state */}
      {loading && (
        <Box
          component={motion.div}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          sx={{
            py: 2,
            display: "flex",
            alignItems: "center",
            gap: 1.5,
          }}
        >
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ repeat: Infinity, duration: 1.2, ease: "linear" }}
          >
            <CircularProgress size={22} />
          </motion.div>
          <Typography sx={{ fontSize: 13, opacity: 0.9 }}>
            Loading pending events…
          </Typography>
        </Box>
      )}

      {!loading && items.length === 0 && (
        <Typography sx={{ mt: 1.5 }}>
          No pending events 🎉
        </Typography>
      )}

      {/* cards */}
      <Stack spacing={2.2} sx={{ mt: 1 }}>
        <AnimatePresence mode="popLayout">
          {items.map((item) => {
            const hasCategory = !!selectedCategory[item.id];
            const hasNotes = !!(notes[item.id] && notes[item.id].trim());
            const disabled = !hasCategory || !hasNotes;

            const biasedScore =
              item.detector?.scores?.biased ?? null;
            const neutralScore =
              item.detector?.scores?.neutral ?? null;

            return (
              <motion.div
                key={item.id}
                layout
                initial={{ opacity: 0, y: 14, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10, scale: 0.96 }}
                transition={{ duration: 0.22, ease: "easeOut" }}
              >
                <Paper
                  sx={{
                    p: 2.2,
                    bgcolor: "rgba(15,23,42,0.98)",
                    borderRadius: 3,
                    border: "1px solid rgba(148,163,184,0.35)",
                    boxShadow: "0 18px 40px rgba(15,23,42,0.9)",
                    position: "relative",
                    overflow: "hidden",
                  }}
                >
                  {/* left gradient accent */}
                  <Box
                    sx={{
                      position: "absolute",
                      insetY: 0,
                      left: 0,
                      width: 4,
                      background:
                        item.source === "user"
                          ? "linear-gradient(to bottom, #fb7185, #f97316)"
                          : "linear-gradient(to bottom, #38bdf8, #a855f7)",
                    }}
                  />

                  <Typography
                    variant="subtitle2"
                    sx={{
                      mb: 0.75,
                      color: "rgba(148,163,184,0.95)",
                      fontSize: 13,
                    }}
                  >
                    <strong>ID:</strong> {item.id}
                    {" • "}
                    <strong>source:</strong> {item.source}
                  </Typography>

                  {/* original / rephrased */}
                  <Typography
                    sx={{
                      mb: 0.5,
                      whiteSpace: "pre-wrap",
                      color: "rgba(248,250,252,0.9)",
                      fontSize: 14,
                    }}
                  >
                    <Box component="span" sx={{ fontWeight: 700 }}>
                      Original:
                    </Box>{" "}
                    {item.preview}
                  </Typography>
                  {item.rephrased && (
                    <Typography
                      sx={{
                        mb: 1,
                        whiteSpace: "pre-wrap",
                        color: "rgba(148,163,184,0.95)",
                        fontSize: 14,
                      }}
                    >
                      <Box component="span" sx={{ fontWeight: 700 }}>
                        Rephrased:
                      </Box>{" "}
                      {item.rephrased}
                    </Typography>
                  )}

                  {/* scores and meter */}
                  {item.detector?.scores && (
                    <Box
                      sx={{
                        mb: 1.2,
                        fontSize: 13,
                        color: "rgba(148,163,184,0.95)",
                      }}
                    >
                      biased:{" "}
                      {biasedScore != null
                        ? biasedScore.toFixed(3)
                        : "—"}{" "}
                      • neutral:{" "}
                      {neutralScore != null
                        ? neutralScore.toFixed(3)
                        : "—"}
                      <Box
                        sx={{
                          mt: 0.75,
                          width: 120,
                          height: 7,
                          borderRadius: 999,
                          overflow: "hidden",
                          border:
                            "1px solid rgba(148,163,184,0.6)",
                          display: "flex",
                          bgcolor: "rgba(15,23,42,0.9)",
                        }}
                      >
                        <Box
                          sx={{
                            width: `${Math.round(
                              (biasedScore || 0) * 100
                            )}%`,
                            transition: "width 280ms ease",
                            background:
                              "linear-gradient(90deg,#fb7185,#f97316)",
                          }}
                        />
                        <Box
                          sx={{
                            flexGrow: 1,
                            background:
                              "linear-gradient(90deg,#22c55e,#22d3ee)",
                          }}
                        />
                      </Box>
                    </Box>
                  )}

                  {/* category and notes */}
                  <Stack
                    direction={{ xs: "column", sm: "row" }}
                    spacing={1}
                    sx={{ mb: 1.3 }}
                    alignItems="stretch"
                  >
                    <TextField
                      select
                      label="Category"
                      size="small"
                      value={selectedCategory[item.id] || ""}
                      onChange={(e) =>
                        setSelectedCategory((prev) => ({
                          ...prev,
                          [item.id]: e.target.value,
                        }))
                      }
                      sx={{ minWidth: 160 }}
                      InputLabelProps={{
                        sx: { color: "rgba(148,163,184,0.9)" },
                      }}
                      InputProps={{
                        sx: {
                          color: "#e5e7eb",
                          "& .MuiOutlinedInput-notchedOutline": {
                            borderColor:
                              "rgba(148,163,184,0.55)",
                          },
                          "&:hover .MuiOutlinedInput-notchedOutline":
                            {
                              borderColor:
                                "rgba(129,140,248,0.9)",
                            },
                          "&.Mui-focused .MuiOutlinedInput-notchedOutline":
                            {
                              borderColor:
                                "rgba(129,140,248,1)",
                            },
                        },
                      }}
                    >
                      {categories.map((c) => (
                        <MenuItem key={c} value={c}>
                          {c}
                        </MenuItem>
                      ))}
                    </TextField>

                    <TextField
                      label="Notes"
                      size="small"
                      fullWidth
                      value={notes[item.id] || ""}
                      onChange={(e) =>
                        setNotes((prev) => ({
                          ...prev,
                          [item.id]: e.target.value,
                        }))
                      }
                      InputLabelProps={{
                        sx: { color: "rgba(148,163,184,0.9)" },
                      }}
                      InputProps={{
                        sx: {
                          color: "#e5e7eb",
                          "& .MuiOutlinedInput-notchedOutline": {
                            borderColor:
                              "rgba(148,163,184,0.55)",
                          },
                          "&:hover .MuiOutlinedInput-notchedOutline":
                            {
                              borderColor:
                                "rgba(129,140,248,0.9)",
                            },
                          "&.Mui-focused .MuiOutlinedInput-notchedOutline":
                            {
                              borderColor:
                                "rgba(129,140,248,1)",
                            },
                        },
                      }}
                    />
                  </Stack>

                  {/* actions with animated press */}
                  <Stack direction="row" spacing={1}>
                    <Button
                      size="small"
                      variant="contained"
                      color="error"
                      disabled={disabled}
                      onClick={() => handleLabel(item.id, true)}
                      component={motion.button}
                      whileTap={
                        disabled ? undefined : { scale: 0.95 }
                      }
                      sx={{
                        textTransform: "none",
                        fontSize: 13,
                        fontWeight: 600,
                        borderRadius: 999,
                        px: 2,
                      }}
                    >
                      Mark as unsafe
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      color="success"
                      disabled={disabled}
                      onClick={() => handleLabel(item.id, false)}
                      component={motion.button}
                      whileTap={
                        disabled ? undefined : { scale: 0.95 }
                      }
                      sx={{
                        textTransform: "none",
                        fontSize: 13,
                        fontWeight: 600,
                        borderRadius: 999,
                        px: 2,
                        borderColor: "rgba(34,197,94,0.8)",
                      }}
                    >
                      Mark as safe
                    </Button>
                    {disabled && (
                      <Typography
                        sx={{
                          ml: 1,
                          fontSize: 11,
                          color: "rgba(248,250,252,0.7)",
                          alignSelf: "center",
                          fontStyle: "italic",
                        }}
                      >
                        Pick a category & notes to enable actions.
                      </Typography>
                    )}
                  </Stack>
                </Paper>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </Stack>
    </Box>
  );
}